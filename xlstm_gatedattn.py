# ============================================================================
# xLSTM VIBRATION PREDICTION - 4-LAYER STACK WITH DEEPSPEED AND CHECKPOINTING
# (Sequence length 100, Batch 4096, LR 0.002, NO attention reasoner)
# MODIFIED: mLSTM with Gated Attention Memory-Augmented Memory (STABILISED)
# ============================================================================
import glob
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC
import matplotlib.pyplot as plt
import os
import math
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from tqdm import tqdm
import time
import json
import deepspeed
import argparse

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================
DATA_FOLDER = '/projects/tc6d/attn_resid/1_stack_gate/Data'
TRAIN_FOLDER = os.path.join(DATA_FOLDER, 'Train')
VAL_FOLDER = os.path.join(DATA_FOLDER, 'Test')
RESULT_FOLDER = os.path.join(DATA_FOLDER, 'Result')
METRICS_FILE = os.path.join(RESULT_FOLDER, 'training_metrics.csv')
CHECKPOINT_FILE = os.path.join(RESULT_FOLDER, 'xlstm_deepspeed_checkpoint.pth')

# Output files for plotting
VAL_LOSS_FILE = os.path.join(RESULT_FOLDER, 'val_loss_history.txt')
VAL_ACCURACY_FILE = os.path.join(RESULT_FOLDER, 'val_accuracy_history.txt')

# Model parameters
SEQUENCE_LENGTH = 100
HIDDEN_SIZE = 128
MEMORY_DIM = 16
NUM_LAYERS = 1  # ← Change this to 1, 2, 3, 4, or 5
NUM_SLOTS = 16  # Number of memory slots for gated attention memory bank
ATTENTION_ITERATIONS = 3  # (kept for config compatibility, but no longer used)

# Training parameters
NUM_EPOCHS = 50
BATCH_SIZE = 4096  # per‑GPU batch size (increased)
LEARNING_RATE = 0.002  # adjusted for larger batch
EARLY_STOP_PATIENCE = 10
SEQUENCE_STEP = 1
USE_MIXED_PRECISION = True

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================
def load_vibration_data_from_folder(folder_path, folder_name="data"):
    """Load all vibration time series data from specified folder"""
    all_series = []
    file_names = []

    print(f"\n{'='*70}")
    print(f"Loading data from {folder_name} folder")
    print(f"Path: {folder_path}")
    print(f"{'='*70}")

    if not os.path.exists(folder_path):
        print(f"❌ ERROR: Folder not found: {folder_path}")
        return all_series, file_names

    csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))

    if len(csv_files) == 0:
        print(f"⚠️  WARNING: No CSV files found in {folder_path}")
        return all_series, file_names

    print(f"Found {len(csv_files)} CSV files\n")

    total_points = 0
    for idx, file in enumerate(csv_files, 1):
        try:
            df = pd.read_csv(file)
            if len(df.columns) >= 2:
                displacement_values = df.iloc[:, 1].values
                all_series.append(displacement_values)
                file_names.append(os.path.basename(file))
                total_points += len(displacement_values)
                print(f"[{idx:2d}/{len(csv_files)}] ✓ {os.path.basename(file):40s} - {len(displacement_values):,} points")
        except Exception as e:
            print(f"[{idx:2d}/{len(csv_files)}] ✗ Error loading {os.path.basename(file)}: {e}")

    print(f"\n{'='*70}")
    print(f"{folder_name} Data Summary:")
    print(f"  Files loaded: {len(all_series)}")
    print(f"  Total data points: {total_points:,}")
    print(f"{'='*70}\n")

    return all_series, file_names


def create_sequences_for_prediction(data, sequence_length=30, step=1):
    """Create sequences for time series prediction"""
    X, y = [], []
    for signal in data:
        if len(signal) > sequence_length:
            for i in range(0, len(signal) - sequence_length, step):
                X.append(signal[i:(i + sequence_length)])
                y.append(signal[i + sequence_length])
    return np.array(X), np.array(y)


def prepare_datasets(train_folder, val_folder, sequence_length=30, step=1):
    """Prepare datasets from separate train and validation folders"""
    start_time = time.time()

    train_series, train_files = load_vibration_data_from_folder(train_folder, "Training")
    val_series, val_files = load_vibration_data_from_folder(val_folder, "Validation")

    if len(train_series) == 0:
        raise ValueError("❌ ERROR: No training data loaded!")
    if len(val_series) == 0:
        raise ValueError("❌ ERROR: No validation data loaded!")

    print(f"Creating sequences with length={sequence_length}, step={step}...\n")

    X_train_list, y_train_list = [], []
    for idx, ts_data in enumerate(train_series, 1):
        X_ts, y_ts = create_sequences_for_prediction([ts_data], sequence_length, step)
        if len(X_ts) > 0:
            X_train_list.extend(X_ts)
            y_train_list.extend(y_ts)
            print(f"  Train file {idx}/{len(train_series)}: {len(X_ts):,} sequences created")

    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)
    print(f"\nTotal training sequences: {len(X_train):,}\n")

    X_val_list, y_val_list = [], []
    for idx, ts_data in enumerate(val_series, 1):
        X_ts, y_ts = create_sequences_for_prediction([ts_data], sequence_length, step)
        if len(X_ts) > 0:
            X_val_list.extend(X_ts)
            y_val_list.extend(y_ts)
            print(f"  Val file {idx}/{len(val_series)}: {len(X_ts):,} sequences created")

    X_val = np.array(X_val_list)
    y_val = np.array(y_val_list)

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Dataset Preparation Complete (took {elapsed:.1f}s)")
    print(f"{'='*70}")
    print(f"Training sequences:  {X_train.shape[0]:,}")
    print(f"Validation sequences: {X_val.shape[0]:,}")
    print(f"Sequence length:      {sequence_length}")
    print(f"{'='*70}\n")

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

    return X_train, X_val, y_train, y_val, train_series, val_series


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
class ImprovedmLSTMBlock(nn.Module):
    """mLSTM block with Gated Attention Memory – stabilised for fp16"""
    def __init__(self, input_size, hidden_size, mem_dim=16, num_slots=16):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mem_dim = mem_dim
        self.num_slots = num_slots
        self.scale = math.sqrt(mem_dim)          # standard attention scaling

        # Original projections
        self.Wq = nn.Linear(input_size, hidden_size)
        self.Wk = nn.Linear(input_size, mem_dim)
        self.Wv = nn.Linear(input_size, mem_dim)
        self.Wi = nn.Linear(input_size + hidden_size, mem_dim)
        self.Wf = nn.Linear(input_size + hidden_size, mem_dim)
        self.Wo = nn.Linear(input_size + mem_dim, hidden_size)

        # Learnable memory bank (smaller init for stability)
        self.memory = nn.Parameter(torch.randn(num_slots, mem_dim) * 0.01)

        # Read/Write query projections with layer norm for stability
        self.read_query_proj = nn.Linear(input_size + hidden_size, mem_dim)
        self.write_query_proj = nn.Linear(input_size + hidden_size, mem_dim)
        self.query_norm = nn.LayerNorm(mem_dim)   # normalise the queries

        # Temperature parameter for attention sharpness (init 1.0)
        self.attn_temp = nn.Parameter(torch.ones(1))

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        for layer in [self.Wq, self.Wk, self.Wv, self.Wi, self.Wf, self.Wo,
                      self.read_query_proj, self.write_query_proj]:
            nn.init.xavier_uniform_(layer.weight, gain=0.5)
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, x, states):
        h_prev, m_prev = states  # m_prev: (batch_size, num_slots, mem_dim)

        # Input normalisation
        x_norm = torch.tanh(x) * 0.3

        # Gate computations
        qt = torch.tanh(self.Wq(x_norm)) * 0.5
        kt = torch.tanh(self.Wk(x_norm)) * (1.0 / math.sqrt(self.mem_dim))
        vt = torch.tanh(self.Wv(x_norm)) * 0.5

        combined = torch.cat([x_norm, torch.tanh(h_prev) * 0.5], dim=1)
        it = torch.sigmoid(self.Wi(combined)) * 0.9 + 0.05  # (B, mem_dim)
        ft = torch.sigmoid(self.Wf(combined)) * 0.9 + 0.05  # (B, mem_dim)

        # ============================================================
        # Write attention (stabilised)
        # ============================================================
        write_q = self.write_query_proj(combined)   # (B, mem_dim)
        write_q = self.query_norm(write_q)          # normalise query

        # Scaled dot-product attention with clamping
        attn_scores = torch.matmul(write_q.unsqueeze(1),
                                   m_prev.transpose(1, 2)) / self.scale
        attn_scores = attn_scores / self.attn_temp.clamp(min=0.1)
        attn_scores = torch.clamp(attn_scores, -20, 20)              # fp16 safety
        attn_weights = torch.softmax(attn_scores, dim=-1)            # (B, 1, num_slots)

        # Write candidate – magnitude-limited
        write_candidate = vt.unsqueeze(1) * attn_weights.squeeze(1).unsqueeze(2)
        write_candidate = torch.tanh(write_candidate)                # keep in [-1, 1]

        # Gated memory update
        it_exp = it.unsqueeze(1).expand(-1, self.num_slots, -1)
        ft_exp = ft.unsqueeze(1).expand(-1, self.num_slots, -1)
        m_t = ft_exp * m_prev + it_exp * write_candidate

        # ============================================================
        # Read attention (stabilised)
        # ============================================================
        read_q = self.read_query_proj(combined)
        read_q = self.query_norm(read_q)

        read_scores = torch.matmul(read_q.unsqueeze(1),
                                   m_t.transpose(1, 2)) / self.scale
        read_scores = read_scores / self.attn_temp.clamp(min=0.1)
        read_scores = torch.clamp(read_scores, -20, 20)
        read_weights = torch.softmax(read_scores, dim=-1)            # (B, 1, num_slots)

        m_t_mean = torch.matmul(read_weights, m_t).squeeze(1)        # (B, mem_dim)

        # ============================================================
        # Output gate
        # ============================================================
        combined_output = torch.cat([x_norm, m_t_mean], dim=1)       # (B, input_size + mem_dim)
        ot = torch.sigmoid(self.Wo(combined_output)) * 0.9 + 0.05
        h_t = ot * torch.tanh(self.layer_norm(qt))
        h_t = self.dropout(h_t)

        return h_t, (h_t, m_t)


class ImprovedsLSTMBlock(nn.Module):
    """sLSTM block for xLSTM"""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_i = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_f = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_c = nn.Linear(input_size + hidden_size, hidden_size)

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        for layer in [self.W_i, self.W_f, self.W_o, self.W_c]:
            nn.init.xavier_uniform_(layer.weight, gain=0.5)
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, x, states):
        h_prev, c_prev = states

        x_norm = torch.tanh(x) * 0.3

        combined = torch.cat([x_norm, torch.tanh(h_prev) * 0.5], dim=1)

        i_t = torch.sigmoid(self.W_i(combined)) * 0.9 + 0.05
        f_t = torch.sigmoid(self.W_f(combined)) * 0.9 + 0.05
        o_t = torch.sigmoid(self.W_o(combined)) * 0.9 + 0.05
        c_hat_t = torch.tanh(self.W_c(combined)) * 0.5

        c_t = f_t * c_prev + i_t * c_hat_t
        h_t = o_t * torch.tanh(self.layer_norm(c_t))
        h_t = self.dropout(h_t)

        return h_t, (h_t, c_t)


class ImprovedxLSTMPredictor(nn.Module):
    """N-Layer Stacked xLSTM with Gated Attention Memory-Augmented mLSTM (no attention reasoner)."""
    def __init__(self, input_size=1, hidden_size=128, mem_dim=16, output_size=1,
                 num_layers=4, attention_iterations=3, num_slots=16):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mem_dim = mem_dim
        self.num_layers = num_layers
        self.num_slots = num_slots

        self.input_proj = nn.Linear(input_size, hidden_size)

        # N mLSTM layers (with Gated Attention Memory)
        self.mlstm_layers = nn.ModuleList([
            ImprovedmLSTMBlock(
                hidden_size if i == 0 else hidden_size // 2,
                hidden_size // 2,
                mem_dim=mem_dim,
                num_slots=num_slots
            ) for i in range(num_layers)
        ])

        # N sLSTM layers
        self.slstm_layers = nn.ModuleList([
            ImprovedsLSTMBlock(hidden_size // 2, hidden_size // 2)
            for i in range(num_layers)
        ])

        # Direct output layer – no attention reasoner
        self.output_layer = nn.Linear(hidden_size // 2, output_size)

        # Layer norms and dropouts between layers
        num_transitions = num_layers * 2 - 1
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size // 2) for _ in range(num_transitions)
        ])
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.1) for _ in range(num_transitions)
        ])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight, gain=0.5)
        nn.init.constant_(self.input_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_layer.weight, gain=0.5)
        nn.init.constant_(self.output_layer.bias, 0.0)

    def init_hidden(self, batch_size, device):
        hidden_states = []

        # mLSTM layers - with slot-based memory
        for i in range(self.num_layers):
            h = torch.zeros(batch_size, self.hidden_size // 2, device=device, dtype=torch.float16)
            m = torch.zeros(batch_size, self.num_slots, self.mem_dim, device=device, dtype=torch.float16)
            hidden_states.append((h, m))

        # sLSTM layers
        for i in range(self.num_layers):
            h = torch.zeros(batch_size, self.hidden_size // 2, device=device, dtype=torch.float16)
            c = torch.zeros(batch_size, self.hidden_size // 2, device=device, dtype=torch.float16)
            hidden_states.append((h, c))

        return hidden_states

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        device = x.device

        hidden_states = self.init_hidden(batch_size, device)

        # Iterate through time steps
        for t in range(seq_len):
            x_t = x[:, t, :]
            x_proj = torch.tanh(self.input_proj(x_t)) * 0.5

            # Process through N mLSTM layers
            for i in range(self.num_layers):
                x_proj, hidden_states[i] = self.mlstm_layers[i](x_proj, hidden_states[i])
                if i < self.num_layers - 1:
                    x_proj = self.layer_norms[i](x_proj)
                    x_proj = self.dropouts[i](x_proj)

            # Process through N sLSTM layers
            for i in range(self.num_layers):
                x_proj, hidden_states[self.num_layers + i] = \
                    self.slstm_layers[i](x_proj, hidden_states[self.num_layers + i])
                if i < self.num_layers - 1:
                    x_proj = self.layer_norms[self.num_layers - 1 + i](x_proj)
                    x_proj = self.dropouts[self.num_layers - 1 + i](x_proj)

        # After the loop, x_proj is the output of the last sLSTM layer at the final time step
        output = self.output_layer(x_proj)
        return output


# ============================================================================
# CHECKPOINTING FUNCTIONS
# ============================================================================
def save_checkpoint(epoch, model, optimizer, scheduler, train_losses, val_losses,
                    val_r2_scores, best_val_loss, patience_counter, training_start_time,
                    filename, args, feature_scaler=None, target_scaler=None):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_r2_scores': val_r2_scores,
        'best_val_loss': best_val_loss,
        'patience_counter': patience_counter,
        'training_start_time': training_start_time,
        'feature_scaler': feature_scaler,
        'target_scaler': target_scaler,
        'config': {
            'SEQUENCE_LENGTH': SEQUENCE_LENGTH,
            'HIDDEN_SIZE': HIDDEN_SIZE,
            'MEMORY_DIM': MEMORY_DIM,
            'NUM_LAYERS': NUM_LAYERS,
            'NUM_SLOTS': NUM_SLOTS,
            'ATTENTION_ITERATIONS': ATTENTION_ITERATIONS,
            'BATCH_SIZE': BATCH_SIZE,
            'LEARNING_RATE': LEARNING_RATE,
            'EARLY_STOP_PATIENCE': EARLY_STOP_PATIENCE,
            'SEQUENCE_STEP': SEQUENCE_STEP,
            'USE_MIXED_PRECISION': USE_MIXED_PRECISION
        }
    }

    if not args.deepspeed or args.local_rank == 0:
        torch.save(checkpoint, filename)
        print(f"✅ Checkpoint saved: {filename}")


def load_checkpoint(filename, model, optimizer, scheduler, device):
    if os.path.isfile(filename):
        print(f"🔄 Loading checkpoint: {filename}")
        checkpoint = torch.load(filename, map_location=device, weights_only=False)

        model.load_state_dict(checkpoint['model_state_dict'])
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except (KeyError, Exception) as e:
            print(f"⚠️  Could not load optimizer state: {e} — skipping")
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return (checkpoint['epoch'],
                checkpoint['train_losses'],
                checkpoint['val_losses'],
                checkpoint['val_r2_scores'],
                checkpoint['best_val_loss'],
                checkpoint['patience_counter'],
                checkpoint['training_start_time'],
                checkpoint.get('feature_scaler', None),
                checkpoint.get('target_scaler', None))
    else:
        print(f"⚠️  No checkpoint found: {filename}")
        return 0, [], [], [], float('inf'), 0, time.time(), None, None


# ============================================================================
# METRICS LOGGING FUNCTIONS
# ============================================================================
def save_metrics_to_csv(epoch_metrics, filename):
    df = pd.DataFrame(epoch_metrics)
    df.to_csv(filename, index=False)
    print(f"✅ Metrics saved to: {filename}")


def save_history_to_txt(epoch, val_loss, val_r2, loss_file, accuracy_file, args):
    if not args.deepspeed or args.local_rank == 0:
        with open(loss_file, 'a') as f:
            f.write(f"{epoch}\t{val_loss:.6f}\n")
        with open(accuracy_file, 'a') as f:
            f.write(f"{epoch}\t{val_r2:.4f}\n")


def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    valid_batches = 0

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            if torch.isnan(batch_X).any() or torch.isinf(batch_X).any():
                continue
            if torch.isnan(batch_y).any() or torch.isinf(batch_y).any():
                continue

            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)

            if not (torch.isnan(loss) or torch.isinf(loss)):
                val_loss += loss.item()
                valid_batches += 1

    return val_loss / max(valid_batches, 1)


def calculate_r2_score(actual, predicted):
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepspeed", action="store_true", help="Enable DeepSpeed")
    parser.add_argument("--deepspeed_config", type=str, help="DeepSpeed config file")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    args = parser.parse_args()

    print("\n" + "="*70)
    print(f" xLSTM VIBRATION PREDICTION - {NUM_LAYERS}-LAYER STACK (GATED ATTENTION MEMORY) (BS={BATCH_SIZE})")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Sequence Length:     {SEQUENCE_LENGTH}")
    print(f"  Hidden Size:         {HIDDEN_SIZE}")
    print(f"  Memory Dim:          {MEMORY_DIM}")
    print(f"  Memory Slots:        {NUM_SLOTS}")
    print(f"  Stacking:            {NUM_LAYERS} mLSTM + {NUM_LAYERS} sLSTM")
    print(f"  Batch Size:          {BATCH_SIZE}")
    print(f"  Learning Rate:       {LEARNING_RATE}")
    print(f"  Max Epochs:          {NUM_EPOCHS}")
    print(f"  Mixed Precision:     {USE_MIXED_PRECISION}")
    print(f"  Stability:           Scaled attention + clamping + query norm + tanh write")
    print("="*70)

    os.makedirs(RESULT_FOLDER, exist_ok=True)

    # Prepare datasets
    X_train, X_val, y_train, y_val, train_series, val_series = prepare_datasets(
        train_folder=TRAIN_FOLDER,
        val_folder=VAL_FOLDER,
        sequence_length=SEQUENCE_LENGTH,
        step=SEQUENCE_STEP
    )

    # Scale data
    print(f"\nScaling data...")
    feature_scaler = RobustScaler()
    target_scaler = RobustScaler()

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)

    X_train_scaled = feature_scaler.fit_transform(X_train_flat).reshape(X_train.shape)
    X_val_scaled = feature_scaler.transform(X_val_flat).reshape(X_val.shape)

    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).flatten()

    print("✅ Data normalized!")

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float16)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float16)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float16)
    y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float16)

    # Build datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    # Initialize model
    model = ImprovedxLSTMPredictor(
        input_size=1,
        hidden_size=HIDDEN_SIZE,
        mem_dim=MEMORY_DIM,
        output_size=1,
        num_layers=NUM_LAYERS,
        attention_iterations=ATTENTION_ITERATIONS,  # ignored, kept for compatibility
        num_slots=NUM_SLOTS
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # DeepSpeed initialization
    if args.deepspeed and args.deepspeed_config:
        print("🚀 Initializing DeepSpeed...")
        model, optimizer, _, _ = deepspeed.initialize(
            args=args,
            model=model,
            optimizer=optimizer
        )
        device = model.local_rank
        print(f"✅ DeepSpeed initialized! Local rank: {device}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

    # DataLoaders
    if args.deepspeed:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        sampler=val_sampler,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"\n{'='*70}")
    print(f" STARTING TRAINING")
    print(f"{'='*70}")
    print(f"Training batches per GPU:   {len(train_loader)}")
    print(f"Validation batches per GPU: {len(val_loader)}")
    print(f"{'='*70}\n")

    # Checkpoint loading
    print(f"\n{'='*70}")
    print(f" CHECKING FOR EXISTING CHECKPOINT")
    print(f"{'='*70}")
    print(f"Checkpoint path: {CHECKPOINT_FILE}")

    map_device = f'cuda:{device}' if isinstance(device, int) else device
    start_epoch, train_losses, val_losses, val_r2_scores, best_val_loss, patience_counter, training_start_time, \
        loaded_feature_scaler, loaded_target_scaler = load_checkpoint(
            CHECKPOINT_FILE, model, optimizer, scheduler, map_device
        )

    if start_epoch > 0:
        print(f"🔄 Resuming from epoch {start_epoch + 1}")
        if loaded_feature_scaler is not None:
            feature_scaler = loaded_feature_scaler
            target_scaler = loaded_target_scaler
            print(f"✅ Loaded scalers from checkpoint")
    else:
        print("🚀 Starting fresh training")
        training_start_time = time.time()
        if not args.deepspeed or args.local_rank == 0:
            if os.path.exists(VAL_LOSS_FILE):
                os.remove(VAL_LOSS_FILE)
            if os.path.exists(VAL_ACCURACY_FILE):
                os.remove(VAL_ACCURACY_FILE)

    print(f"{'='*70}\n")

    epoch_metrics = []
    if start_epoch > 0:
        for i in range(len(train_losses)):
            epoch_metrics.append({
                'epoch': i + 1,
                'train_loss': round(train_losses[i], 6),
                'val_loss': round(val_losses[i], 6),
                'val_r2_score': round(val_r2_scores[i], 4) if i < len(val_r2_scores) else 0.0,
                'epoch_time_sec': 0,
                'learning_rate': LEARNING_RATE
            })
        print(f"✅ Restored {len(epoch_metrics)} epochs of metrics history\n")

    # Training loop
    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_start_time = time.time()

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        epoch_loss = 0
        valid_batches = 0

        progress_bar = tqdm(
            train_loader,
            desc=f'Epoch {epoch+1}/{NUM_EPOCHS}',
            disable=(args.deepspeed and args.local_rank != 0)
        )

        for batch_X, batch_y in progress_bar:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            if torch.isnan(batch_X).any() or torch.isinf(batch_X).any():
                continue
            if torch.isnan(batch_y).any() or torch.isinf(batch_y).any():
                continue

            if not args.deepspeed:
                optimizer.zero_grad()

            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)

            if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 100.0:
                continue

            if args.deepspeed:
                model.backward(loss)
                model.step()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            epoch_loss += loss.item()
            valid_batches += 1

            progress_bar.set_postfix({'Loss': f'{loss.item():.6f}'})

        train_loss = epoch_loss / max(valid_batches, 1)

        # Validation
        val_loss = validate(model, val_loader, criterion, device)

        # R² score
        model.eval()
        val_predictions = []
        val_actuals = []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                val_predictions.extend(outputs.cpu().numpy().flatten())
                val_actuals.extend(batch_y.numpy().flatten())

        val_predictions = target_scaler.inverse_transform(
            np.array(val_predictions).reshape(-1, 1)
        ).flatten()
        val_actuals = target_scaler.inverse_transform(
            np.array(val_actuals).reshape(-1, 1)
        ).flatten()
        val_r2 = calculate_r2_score(val_actuals, val_predictions)

        epoch_time = time.time() - epoch_start_time

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_r2_scores.append(val_r2)

        epoch_metrics.append({
            'epoch': epoch + 1,
            'train_loss': round(train_loss, 6),
            'val_loss': round(val_loss, 6),
            'val_r2_score': round(val_r2, 4),
            'epoch_time_sec': round(epoch_time, 2),
            'learning_rate': optimizer.param_groups[0]['lr']
        })

        if not args.deepspeed or args.local_rank == 0:
            save_metrics_to_csv(epoch_metrics, METRICS_FILE)
            save_history_to_txt(epoch + 1, val_loss, val_r2,
                               VAL_LOSS_FILE, VAL_ACCURACY_FILE, args)

        print(f"\n{'='*70}")
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Time: {epoch_time:.1f}s")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss:   {val_loss:.6f}")
        print(f"  Val R²:     {val_r2:.4f}")
        print(f"  LR:         {optimizer.param_groups[0]['lr']:.6f}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model_path = os.path.join(RESULT_FOLDER, 'best_xlstm_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f"  ✅ New best model saved!")
        else:
            patience_counter += 1
            print(f"  ⏳ Patience: {patience_counter}/{EARLY_STOP_PATIENCE}")

        print(f"{'='*70}")

        save_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_losses=train_losses,
            val_losses=val_losses,
            val_r2_scores=val_r2_scores,
            best_val_loss=best_val_loss,
            patience_counter=patience_counter,
            training_start_time=training_start_time,
            filename=CHECKPOINT_FILE,
            args=args,
            feature_scaler=feature_scaler,
            target_scaler=target_scaler
        )

        if patience_counter >= EARLY_STOP_PATIENCE:
            if not args.deepspeed or args.local_rank == 0:
                print(f"\n🛑 Early stopping triggered at epoch {epoch+1}")
            break

    training_time = time.time() - training_start_time

    # Final evaluation and plotting (only main process)
    if not args.deepspeed or args.local_rank == 0:
        print(f"\n{'='*70}")
        print(f" TRAINING COMPLETED")
        print(f"{'='*70}")
        print(f"Total training time:    {training_time/3600:.2f} hours ({training_time/60:.1f} minutes)")
        print(f"Best validation loss:   {best_val_loss:.6f}")
        print(f"Best validation R²:     {max(val_r2_scores):.4f}")
        print(f"{'='*70}\n")

        save_metrics_to_csv(epoch_metrics, METRICS_FILE)

        model_path = os.path.join(RESULT_FOLDER, 'best_xlstm_model.pth')
        model.load_state_dict(torch.load(model_path, map_location=map_device, weights_only=False))
        print("✅ Loaded best model for evaluation\n")

        print("="*70)
        print(" EVALUATING MODEL")
        print("="*70)

        model.eval()
        predictions = []
        with torch.no_grad():
            for i in tqdm(range(0, len(X_val_tensor), BATCH_SIZE), desc="Evaluating"):
                batch_X = X_val_tensor[i:i+BATCH_SIZE].to(device)
                outputs = model(batch_X)
                predictions.append(outputs.cpu().numpy())

        y_pred_scaled = np.concatenate(predictions).flatten()
        y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_actual = target_scaler.inverse_transform(y_val_scaled.reshape(-1, 1)).flatten()

        valid_mask = np.isfinite(y_actual) & np.isfinite(y_pred)
        y_actual = y_actual[valid_mask]
        y_pred = y_pred[valid_mask]

        mse = mean_squared_error(y_actual, y_pred)
        mae = mean_absolute_error(y_actual, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_actual, y_pred)
        residuals = y_actual - y_pred

        print(f"\n{'='*70}")
        print(f" RESULTS")
        print(f"{'='*70}")
        print(f"\nPerformance Metrics:")
        print(f"  MSE:  {mse:.4f}")
        print(f"  RMSE: {rmse:.4f} nm")
        print(f"  MAE:  {mae:.4f} nm")
        print(f"  R²:   {r2:.4f}")
        print(f"\nTraining Info:")
        print(f"  Total time:     {training_time/3600:.2f} hours")
        print(f"  Epochs completed: {len(train_losses)}")
        print(f"{'='*70}\n")

        save_dict = {
            'model_state_dict': model.state_dict(),
            'feature_scaler': feature_scaler,
            'target_scaler': target_scaler,
            'sequence_length': SEQUENCE_LENGTH,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_r2_scores': val_r2_scores,
            'best_val_loss': best_val_loss,
            'training_time_hours': training_time / 3600,
            'metrics': {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            },
            'config': {
                'hidden_size': HIDDEN_SIZE,
                'memory_dim': MEMORY_DIM,
                'num_layers': NUM_LAYERS,
                'num_slots': NUM_SLOTS,
                'batch_size': BATCH_SIZE,
                'learning_rate': LEARNING_RATE,
                'mixed_precision': USE_MIXED_PRECISION
            }
        }
        torch.save(save_dict, os.path.join(RESULT_FOLDER, 'xlstm_vibration_predictor_final.pth'))
        print(f"✅ Model saved: {os.path.join(RESULT_FOLDER, 'xlstm_vibration_predictor_final.pth')}")

        # Plotting
        fig = plt.figure(figsize=(16, 10))

        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(train_losses, label='Training Loss', linewidth=2)
        ax1.plot(val_losses, label='Validation Loss', linewidth=2)
        ax1.set_title('Training History', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = plt.subplot(2, 3, 2)
        ax2.plot(val_r2_scores, label='Validation R²', linewidth=2, color='green')
        ax2.set_title('Validation R² Score', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('R²')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3 = plt.subplot(2, 3, 3)
        plot_points = min(300, len(y_actual))
        ax3.plot(y_actual[:plot_points], label='Actual', alpha=0.8, linewidth=2)
        ax3.plot(y_pred[:plot_points], label='Predicted', alpha=0.8, linewidth=1.5)
        ax3.set_title('Predictions vs Actual', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Time Steps')
        ax3.set_ylabel('Displacement (nm)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        ax4 = plt.subplot(2, 3, 4)
        ax4.scatter(y_actual, y_pred, alpha=0.4, s=2)
        ax4.plot([y_actual.min(), y_actual.max()],
                 [y_actual.min(), y_actual.max()],
                 'r--', lw=2, label='Perfect')
        ax4.set_xlabel('Actual Displacement (nm)')
        ax4.set_ylabel('Predicted Displacement (nm)')
        ax4.set_title(f'Actual vs Predicted (R² = {r2:.4f})', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        ax5 = plt.subplot(2, 3, 5)
        ax5.hist(residuals, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax5.axvline(0, color='red', linestyle='--', linewidth=2)
        ax5.set_xlabel('Residuals')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Prediction Errors', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)

        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        metrics_text = f"""
PERFORMANCE SUMMARY
-------------------
MSE:  {mse:.4f}
RMSE: {rmse:.4f} nm
MAE:  {mae:.4f} nm
R²:   {r2:.4f}

Training Time: {training_time/3600:.2f}h
Best Val Loss: {best_val_loss:.6f}
Epochs: {len(train_losses)}

Model: {NUM_LAYERS}-Layer xLSTM
       (Gated Attention Memory)
Hidden: {HIDDEN_SIZE}
Memory: {MEMORY_DIM}
Slots:  {NUM_SLOTS}
Batch:  {BATCH_SIZE} | LR: {LEARNING_RATE}
"""
        ax6.text(0.1, 0.5, metrics_text, fontsize=10, family='monospace',
                verticalalignment='center')
        ax6.set_title(f'{NUM_LAYERS}-Layer xLSTM (Gated Attention Memory)',
                      fontsize=11, fontweight='bold')

        plt.tight_layout()
        plot_path = os.path.join(RESULT_FOLDER, 'xlstm_vibration_results.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Results plot saved: {plot_path}")

        print("\n" + "="*70)
        print(" TRAINING AND EVALUATION COMPLETE!")
        print("="*70)
        print(f"\n📁 All results saved to: {RESULT_FOLDER}")
        print(f"   - best_xlstm_model.pth (best model)")
        print(f"   - xlstm_vibration_results.png (visualization)")
        print(f"   - xlstm_vibration_predictor_final.pth (final model)")
        print(f"   - training_metrics.csv (epoch-by-epoch metrics)")
        print(f"   - val_loss_history.txt (for plotting)")
        print(f"   - val_accuracy_history.txt (for plotting)")
        print(f"   - xlstm_deepspeed_checkpoint.pth (checkpoint for resume)")
        print("="*70)