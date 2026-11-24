# ============================================================================
# xLSTM VIBRATION PREDICTION - 4-LAYER STACK WITH DEEPSPEED AND CHECKPOINTING
# WITH KALMAN FILTER PRE-PROCESSING AND MSE LOSS (default)
#
# FIXES APPLIED vs the original KF variant:
#   1. peak_indicator is now CAUSAL — it marks steps AFTER a peak/valley is
#      confirmed by the velocity sign change. (The old version marked the
#      window BEFORE the peak, using future measurements — including the
#      prediction target — as input features: target leakage.)
#   2. PeakAwareLoss threshold (only used with --loss_type peak_aware) is
#      computed from the FULL training-target distribution, not the first
#      batch, so it is deterministic and identical across DeepSpeed ranks.
#   3. Kalman filter covariance P is reset between files (was carrying over
#      from the previous file, causing overconfident transients at file starts).
#   4. NaN-safe data loading (non-numeric/missing cells coerced + interpolated).
#   5. Default loss is plain MSE (--loss_type mse) so val_loss is directly
#      comparable with the base script.
#   6. JERK IS NOW CAUSAL -- this was the remaining target leak. The old
#      `jerk = np.gradient(acceleration, dt)` is a CENTRED difference, so for
#      an input window ending at index i+L-1 its last jerk value was
#      (acc[i+L] - acc[i+L-2])/2, and acc[i+L] is the Kalman posterior after
#      updating with measurement x[i+L] -- the prediction target. The target
#      was algebraically recoverable from the input window, which is why the
#      old kl numbers looked far better than the base model's. It is now a
#      backwards difference: np.diff(acc, prepend=acc[:1]) / dt.
#   7. The savgol pre-smoothing in filter_signal is CENTRED as scipy
#      implements it, so it is delayed by half its window to stay causal if
#      it is ever enabled (it is dead code today: every call passes
#      smoothing_window=None).
#   8. assert_kf_features_causal() runs on the first Train and Validation
#      file before training: perturbing measurement t must not change any
#      feature row before t, and must change at least one row at or after t.
#      A future centred kernel now fails the job instead of inflating the
#      metrics silently.
#   9. The scaler subsample is drawn UNIFORMLY AT RANDOM over the whole
#      training set (seeded, so every rank draws the same indices), instead
#      of taking the first SCALER_SAMPLE_SIZE windows. Windows are
#      concatenated file by file, so the old slice was not a sample of the
#      corpus at all: whenever the alphabetically first file holds more than
#      50,000 windows -- Lathe 316,694 of 5,552,151, synthetic 63,900 of
#      1,533,600 -- every scaler statistic came from that one recording, and
#      the other files landed tens of sigma away. Combined with the
#      `loss > 100` skip that is what made kl/Lathe train on nothing. The
#      scaled training-target range is now printed so a mis-scaled corpus is
#      visible in the log instead of silent.
#
# Folder roles: Train = training | Validation = per-epoch validation |
#   Test = final evaluation only (unseen data)
# ============================================================================
import glob
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
import math
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from tqdm import tqdm
import time
import json
import deepspeed
import argparse
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Kalman filter imports
from filterpy.kalman import KalmanFilter
from scipy import signal as scipy_signal

# ============================================================================
# MATPLOTLIB STYLE
# ============================================================================
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 11
rcParams['axes.linewidth'] = 1.2
rcParams['axes.edgecolor'] = '#333333'
rcParams['axes.grid'] = True
rcParams['grid.alpha'] = 0.4
rcParams['grid.linestyle'] = '--'
rcParams['grid.color'] = '#B0B0B0'
rcParams['xtick.color'] = '#333333'
rcParams['ytick.color'] = '#333333'
rcParams['axes.labelcolor'] = '#333333'
rcParams['text.color'] = '#333333'

COLOR_ACTUAL = '#F09B76'
COLOR_PREDICTED = '#3A3A3A'
COLOR_PERFECT = '#D32F2F'
COLOR_ZERO = '#D32F2F'
COLOR_TREND = '#2E7D32'
COLOR_HIST = '#B0C4DE'

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================
DATA_FOLDER = 'PATH TO DATA FOLDER HERE'
TRAIN_FOLDER = os.path.join(DATA_FOLDER, 'Train')
VAL_FOLDER = os.path.join(DATA_FOLDER, 'Validation')   # FIX: per-epoch validation (was 'Test')
TEST_FOLDER = os.path.join(DATA_FOLDER, 'Test')        # final evaluation only (now truly unseen)
RESULT_FOLDER = os.path.join(DATA_FOLDER, 'Result')
EVAL_RESULT_FOLDER = os.path.join(RESULT_FOLDER, 'evaluation_results')
METRICS_FILE = os.path.join(RESULT_FOLDER, 'training_metrics.csv')
CHECKPOINT_FILE = os.path.join(RESULT_FOLDER, 'xlstm_deepspeed_checkpoint.pth')

# Create evaluation folder at startup
os.makedirs(EVAL_RESULT_FOLDER, exist_ok=True)

# Output files for plotting
VAL_LOSS_FILE = os.path.join(RESULT_FOLDER, 'val_loss_history.txt')
VAL_ACCURACY_FILE = os.path.join(RESULT_FOLDER, 'val_accuracy_history.txt')

# Model parameters
SEQUENCE_LENGTH = 100
HIDDEN_SIZE = 128
MEMORY_DIM = 16
NUM_LAYERS = 1                      # ← Change this to 1, 2, 3, 4, or 5
ATTENTION_ITERATIONS = 3            # (kept for config compatibility, but no longer used)

# Kalman filter parameters
USE_KALMAN_FILTER = True
KF_PROCESS_NOISE = 1e-5           # Trust in dynamics model
KF_MEASUREMENT_NOISE = 1e-3       # Measurement noise level
KF_DT = 1.0                       # Time step
INPUT_FEATURES = 5                # [position, velocity, acceleration, jerk, peak_indicator]
PEAK_LOOKAHEAD = 10               # samples flagged after a confirmed peak/valley

# Loss configuration (only used if --loss_type peak_aware is selected)
PEAK_THRESHOLD_QUANTILE = 0.97

# Training parameters
NUM_EPOCHS = 50
BATCH_SIZE = 4096                   # per‑GPU batch size (increased)
LEARNING_RATE = 0.002               # adjusted for larger batch
EARLY_STOP_PATIENCE = 10
SEQUENCE_STEP = 1
USE_MIXED_PRECISION = False         # float32 throughout

# Memory-efficient parameters
SCALER_SAMPLE_SIZE = 50000          # Number of samples to use for fitting scalers
SCALER_SEED = 0                     # FIX 9: RNG seed for the scaler subsample
SCALING_BATCH_SIZE = 50000          # Batch size for scaling operations

# ============================================================================
# KALMAN FILTER CLASS
# ============================================================================
class VibrationKalmanFilter:
    """
    Kalman Filter for vibration data that provides state estimates [x, v, a]
    where x=displacement, v=velocity, a=acceleration

    This acts as an optimal observer to:
    1. Remove measurement noise
    2. Provide clean velocity and acceleration estimates
    3. Give the model causal peak-context features (velocity goes to zero at peaks)
    """

    def __init__(self,
                 process_noise=1e-4,      # Q - process noise covariance
                 measurement_noise=1e-2,  # R - measurement noise covariance
                 dt=1.0,                   # time step
                 initial_variance=1.0):   # P - initial state covariance
        """
        Initialize Kalman filter for vibration state estimation

        State vector: [x, v, a] (position, velocity, acceleration)
        """
        self.dt = dt

        # FIX 3: store the initial variance so filter_signal can reset P per file
        self.initial_variance = initial_variance

        # Create Kalman filter with 3 states (position, velocity, acceleration)
        self.kf = KalmanFilter(dim_x=3, dim_z=1)

        # State transition matrix (constant acceleration model)
        self.kf.F = np.array([[1, dt, 0.5*dt**2],
                              [0, 1, dt],
                              [0, 0, 1]])

        # Measurement function (we only measure position)
        self.kf.H = np.array([[1, 0, 0]])

        # Process noise covariance
        self.kf.Q = np.eye(3) * process_noise

        # Measurement noise covariance
        self.kf.R = np.array([[measurement_noise]])

        # Initial state covariance
        self.kf.P = np.eye(3) * initial_variance

        # Initial state (position, velocity, acceleration)
        self.kf.x = np.array([[0], [0], [0]])

    def filter_signal(self, displacement_signal, smoothing_window=None):
        """
        Filter a displacement signal and return [x, v, a] states

        Args:
            displacement_signal: Raw displacement measurements
            smoothing_window: Optional additional smoothing

        Returns:
            states: (N, 3) array of [position, velocity, acceleration]
        """
        signal = np.array(displacement_signal, dtype=float)

        # Optional pre-smoothing to reduce high-frequency noise.
        # FIX 7: scipy's savgol_filter is CENTRED -- output t uses samples up
        # to t + window_length//2 -- so the result is delayed by half its
        # window here. The first half window keeps the raw samples, since no
        # causal estimate exists yet. (Dead code as shipped: every call passes
        # smoothing_window=None.)
        if smoothing_window and smoothing_window > 1:
            window_length = min(smoothing_window * 2 + 1, len(signal))
            if window_length >= 5:
                centered = scipy_signal.savgol_filter(
                    signal, window_length=window_length, polyorder=2)
                shift = window_length // 2
                signal = np.concatenate([signal[:shift], centered[:-shift]])

        n_points = len(signal)
        states = np.zeros((n_points, 3))

        # Initialize with first measurement
        self.kf.x = np.array([[signal[0]], [0], [0]])
        # FIX 3: reset the state covariance as well. Without this, P carries
        # over from the end of the PREVIOUS file (this filter object is reused
        # across files in a folder), making the filter overconfident and slow
        # to adapt at the start of every file after the first.
        self.kf.P = np.eye(3) * self.initial_variance

        for i in range(n_points):
            # Predict step
            self.kf.predict()

            # Update step with measurement
            self.kf.update(signal[i])

            # Store state (position, velocity, acceleration)
            states[i, 0] = self.kf.x[0, 0]  # Position
            states[i, 1] = self.kf.x[1, 0]  # Velocity
            states[i, 2] = self.kf.x[2, 0]  # Acceleration

        return states

    def get_peak_detection_features(self, displacement_signal, lookahead=PEAK_LOOKAHEAD):
        """
        Create enhanced features for peak prediction

        Args:
            displacement_signal: Raw displacement signal
            lookahead: Number of steps to mark after a peak/valley is confirmed

        Returns:
            features: (N, 5) array with:
                - position (filtered)
                - velocity
                - acceleration
                - jerk (derivative of acceleration)
                - peak_indicator (CAUSAL flag: +1 for `lookahead` steps AFTER a
                  peak is confirmed by a velocity sign change, -1 after a valley)
        """
        states = self.filter_signal(displacement_signal)

        position = states[:, 0]
        velocity = states[:, 1]
        acceleration = states[:, 2]

        # Calculate jerk (derivative of acceleration).
        # FIX 6: CAUSAL backwards difference. np.gradient is centred, so row t
        # used acceleration[t+1] -- the filter posterior after absorbing
        # measurement x[t+1]. In the last row of a window that is the
        # prediction target itself, which made the target recoverable from the
        # input. prepend= keeps the first sample at 0 instead of reaching
        # into the future.
        jerk = np.diff(acceleration, prepend=acceleration[:1]) / self.dt

        # Create peak indicator (velocity crosses zero from positive to negative = peak)
        n_points = len(position)
        peak_indicator = np.zeros(n_points)

        # Detect when velocity changes sign (potential peaks)
        # FIX 1 (CAUSAL): mark only from the detection point FORWARD. The
        # velocity at step i is computed from measurements up to step i, so
        # marking [i, i+lookahead] uses no future information. The previous
        # version marked [i-lookahead, i], which meant the flag at the last
        # input steps was derived from measurements at/after the prediction
        # target — target leakage.
        velocity_sign = np.sign(velocity)
        for i in range(1, n_points):
            # Detecting when velocity goes from positive to negative (peak)
            if velocity_sign[i-1] > 0 and velocity_sign[i] < 0:
                # Peak confirmed at step i — mark this step and the following window
                end = min(n_points, i + lookahead + 1)
                peak_indicator[i:end] = 1.0
            # Detecting when velocity goes from negative to positive (valley)
            elif velocity_sign[i-1] < 0 and velocity_sign[i] > 0:
                # Valley confirmed at step i — mark this step and the following window
                end = min(n_points, i + lookahead + 1)
                peak_indicator[i:end] = -1.0

        # Combine all features
        features = np.column_stack([position, velocity, acceleration, jerk, peak_indicator])

        return features


# ============================================================================
# NO-LOOKAHEAD SELF-TEST (runs on Train and Validation before every job)
# ============================================================================
def assert_kf_features_causal(series, lookahead=PEAK_LOOKAHEAD, n_probe=20000,
                              rel_tol=1e-11, verbose=True):
    """Perturbing measurement t must not change any feature row before t.

    This is the invariant the legacy pipeline broke. Feature row j must be a
    function of measurements <= j only; `np.gradient` violated it at every
    interior row, and the violation in the LAST row of a window handed the model
    the prediction target.

    Causality is a property of the code, not of the file, so the probe runs on a
    bounded prefix of the series. Rows before t are compared exactly -- a real
    leak moves them by O(1e-2..1) relative, so rel_tol only absorbs floating
    point rounding -- and the rows at or after t must actually move, otherwise
    the probe would be vacuous.
    """
    v = np.asarray(series, dtype=np.float64)[:n_probe]
    if len(v) < 4 * lookahead + 8:
        print(f"  [--] no-lookahead check skipped: series too short ({len(v)} points)")
        return True
    kf = VibrationKalmanFilter(process_noise=KF_PROCESS_NOISE,
                               measurement_noise=KF_MEASUREMENT_NOISE, dt=KF_DT)
    ref = kf.get_peak_detection_features(v, lookahead=lookahead)
    scale = max(1.0, float(np.abs(v).max()))
    step = max(1.0, 1e-4 * scale)          # large enough to be visible downstream
    tol = rel_tol * max(1.0, float(np.abs(ref).max()))
    probes = sorted({lookahead + 3, len(v) // 2, len(v) - 1})
    for t in probes:
        perturbed = v.copy()
        perturbed[t] += step
        new = kf.get_peak_detection_features(perturbed, lookahead=lookahead)
        before = float(np.abs(ref[:t] - new[:t]).max()) if t else 0.0
        assert before <= tol, (
            f"LOOKAHEAD LEAK: perturbing measurement {t} by {step:g} changed feature "
            f"rows BEFORE {t} (max change {before:.6g} > {tol:.6g}); every feature row "
            f"must depend on measurements <= t only")
        after = float(np.abs(ref[t:] - new[t:]).max())
        assert after > 1e-9 * scale, (
            f"the feature builder did not respond to a {step:g} perturbation at {t} "
            f"(max change {after:.6g}) -- the no-lookahead probe would be vacuous")
    if verbose:
        print(f"  [ok] no-lookahead: perturbed {probes} in {len(v):,} points, "
              f"no feature row before t changed")
    return True

# ============================================================================
# CUSTOM LOSS FUNCTIONS
# (default run uses plain MSE; these remain available via --loss_type)
# ============================================================================
class HuberGradientLoss(nn.Module):
    """
    Combined loss function:
    1. Huber Loss (robust to outliers, switch between L1/L2)
    2. Gradient/Derivative Loss (penalizes errors in rate of change)
    3. Asymmetric penalty for under-predicting steep changes
    """
    def __init__(self,
                 huber_delta=1.0,
                 alpha=1.0,
                 beta=0.3,
                 gamma=0.2,
                 underpred_penalty=2.0,
                 gradient_threshold=0.5):
        super().__init__()
        self.huber_delta = huber_delta
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.underpred_penalty = underpred_penalty
        self.gradient_threshold = gradient_threshold

    def gradient_loss(self, y_pred, y_true):
        grad_pred = y_pred[1:] - y_pred[:-1]
        grad_true = y_true[1:] - y_true[:-1]
        return torch.mean(torch.abs(grad_pred - grad_true))

    def asymmetric_loss(self, y_pred, y_true):
        residual = y_pred - y_true
        grad_true = torch.abs(y_true[1:] - y_true[:-1])
        steep_mask = torch.cat([grad_true, grad_true[-1:]])
        steep_mask = (steep_mask > self.gradient_threshold).float()

        underpred_mask = (residual < 0).float()
        overpred_mask = (residual >= 0).float()
        base_loss = torch.abs(residual)

        asymmetric_loss = (
            overpred_mask * base_loss +
            underpred_mask * base_loss * (1.0 + self.underpred_penalty * steep_mask)
        )

        return torch.mean(asymmetric_loss)

    def forward(self, y_pred, y_true):
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        huber_loss = F.smooth_l1_loss(y_pred, y_true, beta=self.huber_delta)
        grad_loss = self.gradient_loss(y_pred, y_true)
        asym_loss = self.asymmetric_loss(y_pred, y_true)

        total_loss = (self.alpha * huber_loss +
                     self.beta * grad_loss +
                     self.gamma * asym_loss)

        self.last_components = {
            'huber': huber_loss.item(),
            'gradient': grad_loss.item(),
            'asymmetric': asym_loss.item(),
            'total': total_loss.item()
        }

        return total_loss

class PeakAwareLoss(nn.Module):
    """
    Loss function that specifically penalizes underprediction of peaks
    """
    def __init__(self,
                 huber_delta=1.0,
                 alpha=1.0,
                 beta=0.3,
                 gamma=0.2,
                 peak_penalty=3.0,
                 peak_threshold_quantile=0.9,
                 underpred_penalty=2.0,
                 gradient_threshold=0.5):
        super().__init__()
        self.huber_delta = huber_delta
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.peak_penalty = peak_penalty
        self.peak_threshold_quantile = peak_threshold_quantile
        self.underpred_penalty = underpred_penalty
        self.gradient_threshold = gradient_threshold
        self.peak_threshold = None

    def calculate_peak_mask(self, y_true):
        """Identify peaks in the target values"""
        if self.peak_threshold is None:
            # Fallback only — main() sets this explicitly from the FULL
            # training-target distribution before training starts (FIX 2)
            self.peak_threshold = np.quantile(y_true.cpu().numpy(), self.peak_threshold_quantile)

        threshold = torch.tensor(self.peak_threshold, device=y_true.device)
        return (y_true > threshold).float()

    def forward(self, y_pred, y_true):
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        # Calculate peak mask
        peak_mask = self.calculate_peak_mask(y_true)

        # 1. Huber Loss
        huber_loss = F.smooth_l1_loss(y_pred, y_true, beta=self.huber_delta)

        # 2. Gradient Loss
        grad_pred = y_pred[1:] - y_pred[:-1]
        grad_true = y_true[1:] - y_true[:-1]
        grad_loss = torch.mean(torch.abs(grad_pred - grad_true))

        # 3. Asymmetric Loss with Peak Penalty
        residual = y_pred - y_true
        underpred_mask = (residual < 0).float()
        overpred_mask = (residual >= 0).float()
        base_loss = torch.abs(residual)

        # Peak-specific penalty
        peak_penalty = peak_mask * underpred_mask * self.peak_penalty

        asym_loss = (
            overpred_mask * base_loss +
            underpred_mask * base_loss * (1.0 + self.underpred_penalty) +
            peak_penalty * base_loss
        )

        # Combine losses
        total_loss = (self.alpha * huber_loss +
                     self.beta * grad_loss +
                     self.gamma * torch.mean(asym_loss))

        # Store for logging
        self.last_components = {
            'huber': huber_loss.item(),
            'gradient': grad_loss.item(),
            'asymmetric': torch.mean(asym_loss).item(),
            'total': total_loss.item(),
            'peak_penalty': peak_penalty.mean().item() if len(peak_penalty) > 0 else 0
        }

        return total_loss

class AdaptiveHuberGradientLoss(nn.Module):
    def __init__(self,
                 huber_delta=1.0,
                 alpha_init=1.0,
                 beta_init=0.1,
                 gamma_init=0.1,
                 underpred_penalty=2.0,
                 gradient_threshold=0.5,
                 total_epochs=50):
        super().__init__()
        self.base_loss = HuberGradientLoss(
            huber_delta=huber_delta,
            alpha=alpha_init,
            beta=beta_init,
            gamma=gamma_init,
            underpred_penalty=underpred_penalty,
            gradient_threshold=gradient_threshold
        )
        self.alpha_init = alpha_init
        self.beta_init = beta_init
        self.gamma_init = gamma_init
        self.total_epochs = total_epochs

    def update_weights(self, current_epoch):
        progress = current_epoch / self.total_epochs
        self.base_loss.alpha = self.alpha_init * (1.0 - 0.5 * progress)
        self.base_loss.beta = self.beta_init * (1.0 + 2.0 * progress)
        self.base_loss.gamma = self.gamma_init * (1.0 + 2.0 * progress)

    def forward(self, y_pred, y_true):
        return self.base_loss(y_pred, y_true)

# ============================================================================
# DATA LOADING AND PREPROCESSING (NaN-safe)
# ============================================================================
def load_vibration_data_with_kf(folder_path, folder_name="data", use_kf=True):
    """Load vibration data with Kalman filter pre-processing"""
    all_series = []
    all_series_kf = []
    file_names = []

    print(f"\n{'='*70}")
    print(f"Loading data from {folder_name} folder {'with Kalman filter' if use_kf else 'without Kalman filter'}")
    print(f"Path: {folder_path}")
    print(f"{'='*70}")

    if not os.path.exists(folder_path):
        print(f"❌ ERROR: Folder not found: {folder_path}")
        return all_series, all_series_kf, file_names

    csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    if len(csv_files) == 0:
        print(f"⚠️ WARNING: No CSV files found in {folder_path}")
        return all_series, all_series_kf, file_names

    print(f"Found {len(csv_files)} CSV files\n")
    total_points = 0

    # Initialize Kalman filter (state + covariance are reset per file inside
    # filter_signal — FIX 3)
    kf = VibrationKalmanFilter(
        process_noise=KF_PROCESS_NOISE,
        measurement_noise=KF_MEASUREMENT_NOISE,
        dt=KF_DT
    )

    for idx, file in enumerate(csv_files, 1):
        try:
            df = pd.read_csv(file)
            if len(df.columns) >= 2:
                # GUARD (FIX 5): coerce non-numeric/missing cells to NaN and
                # interpolate, so one bad cell can't poison training or metrics
                raw_values = pd.to_numeric(df.iloc[:, 1], errors='coerce')
                n_bad = int(raw_values.isna().sum())
                if n_bad > 0:
                    print(f"      ⚠️ {os.path.basename(file)}: {n_bad} NaN/non-numeric values — interpolating")
                    raw_values = raw_values.interpolate(limit_direction='both').ffill().bfill()
                displacement_values = raw_values.to_numpy(dtype=np.float64)
                if np.isnan(displacement_values).any():
                    displacement_values = np.nan_to_num(displacement_values, nan=0.0)

                all_series.append(displacement_values)
                file_names.append(os.path.basename(file))
                total_points += len(displacement_values)

                if use_kf:
                    # Apply Kalman filter to get multi-channel features
                    kf_features = kf.get_peak_detection_features(
                        displacement_values, lookahead=PEAK_LOOKAHEAD
                    )
                    all_series_kf.append(kf_features)
                    print(f"[{idx:2d}/{len(csv_files)}] ✓ {os.path.basename(file):40s} - {len(displacement_values):,} points (KF applied)")
                else:
                    print(f"[{idx:2d}/{len(csv_files)}] ✓ {os.path.basename(file):40s} - {len(displacement_values):,} points")

        except Exception as e:
            print(f"[{idx:2d}/{len(csv_files)}] ✗ Error loading {os.path.basename(file)}: {e}")

    print(f"\n{'='*70}")
    print(f"{folder_name} Data Summary:")
    print(f"  Files loaded: {len(all_series)}")
    print(f"  Total data points: {total_points:,}")
    if use_kf:
        print(f"  Kalman filter applied: Yes (causal peak indicators)")
        print(f"  Input features: {INPUT_FEATURES} channels")
    print(f"{'='*70}\n")

    return all_series, all_series_kf, file_names

def create_sequences_with_kf_features(data_kf, target_data, sequence_length=30, step=1):
    """Create sequences using Kalman filter features as input"""
    X, y = [], []

    for kf_signal, target_signal in zip(data_kf, target_data):
        if len(kf_signal) > sequence_length:
            for i in range(0, len(kf_signal) - sequence_length, step):
                # Input: KF features [x, v, a, jerk, peak_indicator]
                X.append(kf_signal[i:(i + sequence_length)])
                # Target: next displacement value
                y.append(target_signal[i + sequence_length])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def create_sequences_for_prediction(data, sequence_length=30, step=1):
    """Create sequences for time series prediction (single channel)"""
    X, y = [], []
    for signal in data:
        if len(signal) > sequence_length:
            for i in range(0, len(signal) - sequence_length, step):
                X.append(signal[i:(i + sequence_length)])
                y.append(signal[i + sequence_length])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# ============================================================================
# MEMORY-EFFICIENT DATASET PREPARATION WITH KALMAN FILTER
# ============================================================================
def prepare_datasets_with_kf(train_folder, val_folder, sequence_length=30, step=1, use_kf=True):
    """Prepare datasets using Kalman filter with memory efficiency"""
    start_time = time.time()

    # Load with Kalman filter
    train_series, train_series_kf, train_files = load_vibration_data_with_kf(
        train_folder, "Training", use_kf=use_kf
    )
    val_series, val_series_kf, val_files = load_vibration_data_with_kf(
        val_folder, "Validation", use_kf=use_kf
    )

    if len(train_series) == 0:
        raise ValueError("❌ ERROR: No training data loaded!")
    if len(val_series) == 0:
        raise ValueError("❌ ERROR: No validation data loaded!")

    print(f"Creating sequences with length={sequence_length}, step={step}...\n")

    # Process training data with float32 immediately
    X_train_list = []
    y_train_list = []

    for idx, (ts_data, kf_data) in enumerate(zip(train_series, train_series_kf), 1):
        if use_kf:
            X_ts, y_ts = create_sequences_with_kf_features([kf_data], [ts_data], sequence_length, step)
        else:
            X_ts, y_ts = create_sequences_for_prediction([ts_data], sequence_length, step)

        if len(X_ts) > 0:
            # Convert to float32 IMMEDIATELY to save memory
            X_ts = X_ts.astype(np.float32)
            y_ts = y_ts.astype(np.float32)

            X_train_list.append(X_ts)
            y_train_list.append(y_ts)
            print(f"  Train file {idx}/{len(train_series)}: {len(X_ts):,} sequences created")

    # Concatenate with float32 (memory efficient)
    if X_train_list:
        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
    else:
        X_train = np.array([], dtype=np.float32)
        y_train = np.array([], dtype=np.float32)

    # Clear memory
    del X_train_list, y_train_list

    print(f"\nTotal training sequences: {len(X_train):,}\n")

    # Process validation data with float32 immediately
    X_val_list = []
    y_val_list = []

    for idx, (ts_data, kf_data) in enumerate(zip(val_series, val_series_kf), 1):
        if use_kf:
            X_ts, y_ts = create_sequences_with_kf_features([kf_data], [ts_data], sequence_length, step)
        else:
            X_ts, y_ts = create_sequences_for_prediction([ts_data], sequence_length, step)

        if len(X_ts) > 0:
            # Convert to float32 IMMEDIATELY to save memory
            X_ts = X_ts.astype(np.float32)
            y_ts = y_ts.astype(np.float32)

            X_val_list.append(X_ts)
            y_val_list.append(y_ts)
            print(f"  Val file {idx}/{len(val_series)}: {len(X_ts):,} sequences created")

    # Concatenate with float32 (memory efficient)
    if X_val_list:
        X_val = np.concatenate(X_val_list, axis=0)
        y_val = np.concatenate(y_val_list, axis=0)
    else:
        X_val = np.array([], dtype=np.float32)
        y_val = np.array([], dtype=np.float32)

    # Clear memory
    del X_val_list, y_val_list

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Dataset Preparation Complete (took {elapsed:.1f}s)")
    print(f"{'='*70}")
    print(f"Training sequences: {X_train.shape[0]:,}")
    print(f"Validation sequences: {X_val.shape[0]:,}")
    if use_kf:
        print(f"Input features: {X_train.shape[2]}")
    else:
        print(f"Input features: 1")
    print(f"Sequence length: {sequence_length}")
    print(f"{'='*70}\n")

    return X_train, X_val, y_train, y_val, train_series, val_series

# ============================================================================
# MEMORY-EFFICIENT SCALING FUNCTION (IN-PLACE)
# ============================================================================
def scale_data_memory_efficient(X_train, X_val, y_train, y_val, feature_scaler, target_scaler, use_kf=True):
    """Scale data in batches to avoid memory issues - processes in-place"""

    print("Scaling data in batches...")

    # Scale training data in-place (overwrite original to save memory)
    batch_size = SCALING_BATCH_SIZE

    print("Scaling training data...")
    for i in tqdm(range(0, len(X_train), batch_size), desc="Scaling train"):
        end = min(i + batch_size, len(X_train))
        batch_flat = X_train[i:end].reshape(end - i, -1)
        batch_scaled = feature_scaler.transform(batch_flat.astype(np.float32))

        if use_kf:
            X_train[i:end] = batch_scaled.reshape(end - i, X_train.shape[1], X_train.shape[2])
        else:
            X_train[i:end] = batch_scaled.reshape(end - i, X_train.shape[1], 1)

        del batch_flat, batch_scaled

    print("✅ Training data scaled")

    # Scale validation data in-place (overwrite original to save memory)
    print("Scaling validation data...")
    for i in tqdm(range(0, len(X_val), batch_size), desc="Scaling val"):
        end = min(i + batch_size, len(X_val))
        batch_flat = X_val[i:end].reshape(end - i, -1)
        batch_scaled = feature_scaler.transform(batch_flat.astype(np.float32))

        if use_kf:
            X_val[i:end] = batch_scaled.reshape(end - i, X_val.shape[1], X_val.shape[2])
        else:
            X_val[i:end] = batch_scaled.reshape(end - i, X_val.shape[1], 1)

        del batch_flat, batch_scaled

    print("✅ Validation data scaled")

    # Scale targets (these are smaller - 1D)
    print("Scaling targets...")
    y_train_scaled = target_scaler.transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).flatten()

    print("✅ Targets scaled")

    # Return the in-place scaled arrays
    return X_train, X_val, y_train_scaled, y_val_scaled

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
class ImprovedmLSTMBlock(nn.Module):
    """mLSTM block for xLSTM"""
    def __init__(self, input_size, hidden_size, mem_dim=16):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mem_dim = mem_dim

        self.Wq = nn.Linear(input_size, hidden_size)
        self.Wk = nn.Linear(input_size, mem_dim)
        self.Wv = nn.Linear(input_size, mem_dim)
        self.Wi = nn.Linear(input_size + hidden_size, mem_dim)
        self.Wf = nn.Linear(input_size + hidden_size, mem_dim)
        self.Wo = nn.Linear(input_size + 1, hidden_size)

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)
        self._init_weights()

    def _init_weights(self):
        for layer in [self.Wq, self.Wk, self.Wv, self.Wi, self.Wf, self.Wo]:
            nn.init.xavier_uniform_(layer.weight, gain=0.5)
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, x, states):
        h_prev, m_prev = states

        x_norm = x
        qt = torch.tanh(self.Wq(x_norm)) * 0.5
        kt = torch.tanh(self.Wk(x_norm)) * (1.0 / math.sqrt(self.mem_dim))
        vt = torch.tanh(self.Wv(x_norm)) * 0.5

        combined = torch.cat([x_norm, torch.tanh(h_prev) * 0.5], dim=1)
        it = torch.sigmoid(self.Wi(combined)) * 0.9 + 0.05
        ft = torch.sigmoid(self.Wf(combined)) * 0.9 + 0.05

        vt_kt_outer = torch.bmm(vt.unsqueeze(2), kt.unsqueeze(1)) * 0.1
        it_expanded = it.unsqueeze(2).expand_as(vt_kt_outer)
        ft_expanded = ft.unsqueeze(2).expand_as(vt_kt_outer)
        m_t = ft_expanded * m_prev + it_expanded * vt_kt_outer

        m_t_mean = m_t.mean(dim=(1, 2), keepdim=False)
        combined_output = torch.cat([x_norm, m_t_mean.unsqueeze(1)], dim=1)
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

        x_norm = x
        combined = torch.cat([x_norm, torch.tanh(h_prev) * 0.5], dim=1)

        i_t = torch.sigmoid(self.W_i(combined)) * 0.9 + 0.05
        f_t = torch.sigmoid(self.W_f(combined)) * 0.9 + 0.05
        o_t = torch.sigmoid(self.W_o(combined)) * 0.9 + 0.05
        c_hat_t = torch.tanh(self.W_c(combined)) * 0.5

        c_t = f_t * c_prev + i_t * c_hat_t
        h_t = o_t * torch.tanh(self.layer_norm(c_t))
        h_t = self.dropout(h_t)

        return h_t, (h_t, c_t)

class ImprovedxLSTMPredictor_MultiChannel(nn.Module):
    """N-Layer Stacked xLSTM with multi-channel input support"""
    def __init__(self, input_size=5, hidden_size=128, mem_dim=16,
                 output_size=1, num_layers=4, attention_iterations=3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mem_dim = mem_dim
        self.num_layers = num_layers

        # Input projection handles multiple channels
        self.input_proj = nn.Linear(input_size, hidden_size)

        # N mLSTM layers
        self.mlstm_layers = nn.ModuleList([
            ImprovedmLSTMBlock(hidden_size if i == 0 else hidden_size // 2,
                               hidden_size // 2,
                               mem_dim=mem_dim)
            for i in range(num_layers)
        ])

        # N sLSTM layers
        self.slstm_layers = nn.ModuleList([
            ImprovedsLSTMBlock(hidden_size // 2, hidden_size // 2)
            for _ in range(num_layers)
        ])

        # Direct output layer
        self.output_layer = nn.Linear(hidden_size // 2, output_size)

        # Layer norms and dropouts between layers
        num_transitions = num_layers * 2 - 1
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_size // 2) for _ in range(num_transitions)])
        self.dropouts = nn.ModuleList([nn.Dropout(0.1) for _ in range(num_transitions)])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight, gain=0.5)
        nn.init.constant_(self.input_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_layer.weight, gain=0.5)
        nn.init.constant_(self.output_layer.bias, 0.0)

    def init_hidden(self, batch_size, device):
        hidden_states = []
        for i in range(self.num_layers):
            h = torch.zeros(batch_size, self.hidden_size // 2, device=device)
            m = torch.zeros(batch_size, self.mem_dim, self.mem_dim, device=device)
            hidden_states.append((h, m))
        for i in range(self.num_layers):
            h = torch.zeros(batch_size, self.hidden_size // 2, device=device)
            c = torch.zeros(batch_size, self.hidden_size // 2, device=device)
            hidden_states.append((h, c))
        return hidden_states

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        device = x.device
        hidden_states = self.init_hidden(batch_size, device)

        for t in range(seq_len):
            x_t = x[:, t, :]
            x_proj = self.input_proj(x_t)

            for i in range(self.num_layers):
                x_proj, hidden_states[i] = self.mlstm_layers[i](x_proj, hidden_states[i])
                if i < self.num_layers - 1:
                    x_proj = self.layer_norms[i](x_proj)
                    x_proj = self.dropouts[i](x_proj)

            for i in range(self.num_layers):
                x_proj, hidden_states[self.num_layers + i] = self.slstm_layers[i](x_proj, hidden_states[self.num_layers + i])
                if i < self.num_layers - 1:
                    x_proj = self.layer_norms[self.num_layers - 1 + i](x_proj)
                    x_proj = self.dropouts[self.num_layers - 1 + i](x_proj)

        output = self.output_layer(x_proj)
        return output

# ============================================================================
# SAFETY FUNCTIONS
# ============================================================================
def clean_array(arr, name="array"):
    """Clean array by removing/replacing NaN, Inf, and extreme values"""
    arr = np.array(arr, dtype=np.float64)

    nan_count = np.sum(np.isnan(arr))
    inf_count = np.sum(np.isinf(arr))

    if nan_count > 0 or inf_count > 0:
        print(f"  ⚠️ {name}: Found {nan_count} NaN and {inf_count} Inf values")

    arr[np.isinf(arr)] = np.nan

    if np.sum(~np.isnan(arr)) > 0:
        median_val = np.nanmedian(arr)
        arr[np.isnan(arr)] = median_val
    else:
        print(f"  ❌ {name}: All values are NaN!")
        arr = np.zeros_like(arr)

    if len(arr) > 0:
        median = np.median(arr)
        std = np.std(arr)
        if std > 0 and std < 1e10:
            lower = median - 50 * std
            upper = median + 50 * std
            arr = np.clip(arr, lower, upper)

    return arr

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================
def setup_plot_style(ax):
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for spine in ax.spines.values():
        spine.set_color('#333333')
        spine.set_linewidth(1.2)
    ax.grid(True, linestyle='--', alpha=0.4, color='#B0B0B0')
    ax.tick_params(colors='#333333')
    ax.set_facecolor('white')

def plot_predictions_per_file(filename, y_pred, y_actual, save_dir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor('white')
    fig.suptitle(f'Predictions for {filename}', fontsize=14, fontweight='bold', color='#333333')

    y_actual = clean_array(y_actual, "actual")
    y_pred = clean_array(y_pred, "predicted")

    n_points = min(500, len(y_pred))
    ax1 = axes[0, 0]
    ax1.plot(range(n_points), y_actual[:n_points], color=COLOR_ACTUAL, label='Actual',
             linewidth=1.8, alpha=0.9)
    ax1.plot(range(n_points), y_pred[:n_points], color=COLOR_PREDICTED, label='Predicted',
             linewidth=1.5, alpha=0.9, linestyle='--')
    ax1.set_xlabel('Time Step', fontweight='bold')
    ax1.set_ylabel('Displacement (nm)', fontweight='bold')
    ax1.set_title(f'First {n_points} Time Steps - Predicted vs Actual', fontsize=12, fontweight='bold')
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    setup_plot_style(ax1)

    ax2 = axes[0, 1]
    ax2.plot(range(len(y_actual)), y_actual, color=COLOR_ACTUAL, label='Actual',
             linewidth=1.3, alpha=0.8)
    ax2.plot(range(len(y_pred)), y_pred, color=COLOR_PREDICTED, label='Predicted',
             linewidth=1.0, alpha=0.8, linestyle='--')
    ax2.set_xlabel('Time Step', fontweight='bold')
    ax2.set_ylabel('Displacement (nm)', fontweight='bold')
    ax2.set_title(f'Full Time Series ({len(y_actual):,} points)', fontsize=12, fontweight='bold')
    ax2.legend(frameon=True, fancybox=True, shadow=True)
    setup_plot_style(ax2)

    ax3 = axes[1, 0]
    ax3.scatter(y_actual, y_pred, alpha=0.4, s=8, c=COLOR_ACTUAL, edgecolors='white', linewidth=0.5)
    min_val = min(y_actual.min(), y_pred.min())
    max_val = max(y_actual.max(), y_pred.max())
    ax3.plot([min_val, max_val], [min_val, max_val], color=COLOR_PERFECT, linestyle='--', lw=2,
             label='Perfect Prediction')
    file_r2 = r2_score(y_actual, y_pred)
    ax3.set_xlabel('Actual Displacement (nm)', fontweight='bold')
    ax3.set_ylabel('Predicted Displacement (nm)', fontweight='bold')
    ax3.set_title(f'Actual vs Predicted (R² = {file_r2:.4f})', fontsize=12, fontweight='bold')
    ax3.legend(frameon=True, fancybox=True, shadow=True)
    setup_plot_style(ax3)

    ax4 = axes[1, 1]
    residuals = y_actual - y_pred
    ax4.plot(range(len(residuals)), residuals, color=COLOR_PREDICTED, linewidth=0.8, alpha=0.7)
    ax4.axhline(y=0, color=COLOR_ZERO, linestyle='--', linewidth=1.5)
    ax4.set_xlabel('Time Step', fontweight='bold')
    ax4.set_ylabel('Residual (nm)', fontweight='bold')
    ax4.set_title('Residuals Over Time', fontsize=12, fontweight='bold')
    setup_plot_style(ax4)

    plt.tight_layout()
    safe_name = filename.replace('.csv', '').replace(' ', '_')
    save_path = os.path.join(save_dir, f'predictions_{safe_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    return save_path

def plot_combined_residuals(all_y_actual, all_y_pred, save_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.patch.set_facecolor('white')
    fig.suptitle('Combined Residual Analysis - All Test Files',
                 fontsize=14, fontweight='bold', color='#333333')

    all_y_actual = clean_array(all_y_actual, "combined actual")
    all_y_pred = clean_array(all_y_pred, "combined predicted")

    residuals = all_y_actual - all_y_pred
    abs_residuals = np.abs(residuals)

    ax1 = axes[0, 0]
    ax1.scatter(all_y_pred, residuals, alpha=0.3, s=5, c=COLOR_ACTUAL, edgecolors='white', linewidth=0.3)
    ax1.axhline(y=0, color=COLOR_ZERO, linestyle='--', linewidth=2)
    ax1.set_xlabel('Predicted Displacement (nm)', fontweight='bold')
    ax1.set_ylabel('Residual (nm)', fontweight='bold')
    ax1.set_title('Residuals vs Predicted Values', fontsize=12, fontweight='bold')
    try:
        z = np.polyfit(all_y_pred, residuals, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(all_y_pred.min(), all_y_pred.max(), 100)
        ax1.plot(x_trend, p(x_trend), color=COLOR_TREND, linewidth=2.5,
                 label=f'Trend (slope={z[0]:.6f})')
        ax1.legend(frameon=True, fancybox=True, shadow=True)
    except:
        pass
    setup_plot_style(ax1)

    ax2 = axes[0, 1]
    ax2.hist(residuals, bins=100, alpha=0.7, color=COLOR_HIST, edgecolor='#3A3A3A',
             linewidth=0.5, density=True)
    ax2.axvline(x=0, color=COLOR_ZERO, linestyle='--', linewidth=2)
    try:
        mu, std = stats.norm.fit(residuals)
        x = np.linspace(residuals.min(), residuals.max(), 100)
        p = stats.norm.pdf(x, mu, std)
        ax2.plot(x, p, color=COLOR_PERFECT, linewidth=2.5,
                 label=f'Normal Fit (μ={mu:.4f}, σ={std:.4f})')
        ax2.legend(frameon=True, fancybox=True, shadow=True)
    except:
        pass
    ax2.set_xlabel('Residual (nm)', fontweight='bold')
    ax2.set_ylabel('Density', fontweight='bold')
    ax2.set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
    setup_plot_style(ax2)

    ax3 = axes[1, 0]
    try:
        stats.probplot(residuals, dist="norm", plot=ax3)
        ax3.get_lines()[0].set_markerfacecolor(COLOR_ACTUAL)
        ax3.get_lines()[0].set_markeredgecolor('#3A3A3A')
        ax3.get_lines()[0].set_alpha(0.6)
        ax3.get_lines()[1].set_color(COLOR_PERFECT)
        ax3.get_lines()[1].set_linewidth(2)
    except:
        ax3.text(0.5, 0.5, 'Q-Q Plot unavailable', ha='center', va='center', transform=ax3.transAxes)
    ax3.set_title('Q-Q Plot of Residuals', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Theoretical Quantiles', fontweight='bold')
    ax3.set_ylabel('Sample Quantiles', fontweight='bold')
    setup_plot_style(ax3)

    ax4 = axes[1, 1]
    ax4.scatter(all_y_pred, abs_residuals, alpha=0.3, s=5, c=COLOR_ACTUAL, edgecolors='white', linewidth=0.3)
    ax4.set_xlabel('Predicted Displacement (nm)', fontweight='bold')
    ax4.set_ylabel('Absolute Residual (nm)', fontweight='bold')
    ax4.set_title('Absolute Residuals vs Predicted Values', fontsize=12, fontweight='bold')
    try:
        z = np.polyfit(all_y_pred, abs_residuals, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(all_y_pred.min(), all_y_pred.max(), 100)
        ax4.plot(x_trend, p(x_trend), color=COLOR_TREND, linewidth=2.5,
                 label=f'Trend (slope={z[0]:.6f})')
        ax4.legend(frameon=True, fancybox=True, shadow=True)
    except:
        pass
    setup_plot_style(ax4)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'combined_residual_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    return save_path

# ============================================================================
# INTEGRATED EVALUATION FUNCTION
# ============================================================================
def run_full_evaluation(model, feature_scaler, target_scaler, device, args):
    """Run the complete evaluation"""

    print(f"\n{'='*70}")
    print(f"  RUNNING INTEGRATED EVALUATION")
    print(f"{'='*70}")

    os.makedirs(EVAL_RESULT_FOLDER, exist_ok=True)
    print(f"Evaluation results will be saved to: {EVAL_RESULT_FOLDER}")

    # Load test data
    test_series, test_series_kf, test_files = load_vibration_data_with_kf(
        TEST_FOLDER, "Test", use_kf=USE_KALMAN_FILTER
    )

    if len(test_series) == 0:
        print("❌ ERROR: No test data found!")
        return None

    print(f"✅ Loaded {len(test_series)} test files")

    # Get model configuration
    if hasattr(model, 'module'):
        hidden_size = model.module.hidden_size if hasattr(model.module, 'hidden_size') else HIDDEN_SIZE
        num_layers = model.module.num_layers if hasattr(model.module, 'num_layers') else NUM_LAYERS
    else:
        hidden_size = model.hidden_size if hasattr(model, 'hidden_size') else HIDDEN_SIZE
        num_layers = model.num_layers if hasattr(model, 'num_layers') else NUM_LAYERS

    print(f"\nModel Configuration for Evaluation:")
    print(f"  Sequence Length: {SEQUENCE_LENGTH}")
    print(f"  Hidden Size: {hidden_size}")
    print(f"  Memory Dim: {MEMORY_DIM}")
    print(f"  Num Layers: {num_layers}")

    # METHOD 1: Exact replica of training evaluation
    print(f"\n{'='*70}")
    print(f" METHOD 1: Exact Training Evaluation")
    print(f"{'='*70}")

    X_test_list, y_test_list = [], []
    if USE_KALMAN_FILTER:
        for kf_data, ts_data in zip(test_series_kf, test_series):
            X_ts, y_ts = create_sequences_with_kf_features([kf_data], [ts_data], SEQUENCE_LENGTH, SEQUENCE_STEP)
            if len(X_ts) > 0:
                X_test_list.extend(X_ts)
                y_test_list.extend(y_ts)
    else:
        for ts_data in test_series:
            X_ts, y_ts = create_sequences_for_prediction([ts_data], SEQUENCE_LENGTH, SEQUENCE_STEP)
            if len(X_ts) > 0:
                X_test_list.extend(X_ts)
                y_test_list.extend(y_ts)

    X_test = np.array(X_test_list, dtype=np.float32)
    y_test = np.array(y_test_list, dtype=np.float32)
    print(f"Total sequences: {len(X_test):,}")

    if len(X_test) == 0:
        print("❌ ERROR: No test sequences could be created (files too short?)")
        return None

    # Scale
    if USE_KALMAN_FILTER:
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
    else:
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)

    X_test_scaled = feature_scaler.transform(X_test_flat.astype(np.float32))
    if USE_KALMAN_FILTER:
        X_test_scaled = X_test_scaled.reshape(X_test.shape[0], X_test.shape[1], INPUT_FEATURES)
    else:
        X_test_scaled = X_test_scaled.reshape(X_test.shape[0], X_test.shape[1], 1)

    y_test_scaled = target_scaler.transform(y_test.reshape(-1, 1)).flatten()

    print(f"\n📊 Scaled data stats:")
    print(f"  X_scaled range: [{X_test_scaled.min():.4f}, {X_test_scaled.max():.4f}]")
    print(f"  y_scaled range: [{y_test_scaled.min():.4f}, {y_test_scaled.max():.4f}]")

    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    model.eval()
    predictions = []

    with torch.no_grad():
        for i in tqdm(range(0, len(X_test_tensor), BATCH_SIZE), desc="Evaluating"):
            batch_X = X_test_tensor[i:i+BATCH_SIZE].to(device)
            outputs = model(batch_X)
            pred_batch = outputs.cpu().numpy()
            predictions.append(pred_batch)

    # NOTE: clean_array converts to float64 before inverse_transform — safe
    y_pred_scaled = np.concatenate(predictions).flatten()
    y_pred_scaled = clean_array(y_pred_scaled, "scaled predictions")

    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_actual = target_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()

    y_pred = clean_array(y_pred, "final predictions")
    y_actual = clean_array(y_actual, "final actual")

    print(f"\n📊 Final data stats:")
    print(f"  Actual range: [{y_actual.min():.4f}, {y_actual.max():.4f}]")
    print(f"  Predicted range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")

    mse = mean_squared_error(y_actual, y_pred)
    mae = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_actual, y_pred)

    print(f"\n📈 Overall Metrics:")
    print(f"  MSE:  {mse:.6f}")
    print(f"  RMSE: {rmse:.4f} nm")
    print(f"  MAE:  {mae:.4f} nm")
    print(f"  R²:   {r2:.4f}")

    # METHOD 2: Per-file evaluation with plots
    print(f"\n{'='*70}")
    print(f" METHOD 2: Per-File Evaluation")
    print(f"{'='*70}\n")

    all_y_actual_files = []
    all_y_pred_files = []
    per_file_metrics = {}

    for filename, series, kf_data in zip(test_files, test_series, test_series_kf):
        print(f"Evaluating: {filename}")

        if USE_KALMAN_FILTER:
            X_file, y_file = create_sequences_with_kf_features([kf_data], [series], SEQUENCE_LENGTH, SEQUENCE_STEP)
        else:
            X_file, y_file = create_sequences_for_prediction([series], SEQUENCE_LENGTH, SEQUENCE_STEP)

        if len(X_file) == 0:
            print(f"  ⚠️ Skipped (too few points)")
            continue

        if USE_KALMAN_FILTER:
            X_file_flat = X_file.reshape(X_file.shape[0], -1)
            X_file_scaled = feature_scaler.transform(X_file_flat.astype(np.float32))
            X_file_scaled = X_file_scaled.reshape(X_file.shape[0], X_file.shape[1], INPUT_FEATURES)
        else:
            X_file = X_file.reshape(X_file.shape[0], X_file.shape[1], 1)
            X_file_flat = X_file.reshape(X_file.shape[0], -1)
            X_file_scaled = feature_scaler.transform(X_file_flat.astype(np.float32)).reshape(X_file.shape)

        y_file_scaled = target_scaler.transform(y_file.reshape(-1, 1)).flatten()

        X_file_tensor = torch.tensor(X_file_scaled, dtype=torch.float32)
        file_predictions = []

        with torch.no_grad():
            for i in range(0, len(X_file_tensor), BATCH_SIZE):
                batch_X = X_file_tensor[i:i+BATCH_SIZE].to(device)
                outputs = model(batch_X)
                file_predictions.append(outputs.cpu().numpy())

        # NOTE: clean_array converts to float64 before inverse_transform — safe
        y_pred_file_scaled = np.concatenate(file_predictions).flatten()
        y_pred_file_scaled = clean_array(y_pred_file_scaled, f"{filename} scaled pred")

        y_pred_file = target_scaler.inverse_transform(y_pred_file_scaled.reshape(-1, 1)).flatten()
        y_actual_file = target_scaler.inverse_transform(y_file_scaled.reshape(-1, 1)).flatten()

        y_pred_file = clean_array(y_pred_file, f"{filename} pred")
        y_actual_file = clean_array(y_actual_file, f"{filename} actual")

        file_mse = mean_squared_error(y_actual_file, y_pred_file)
        file_rmse = np.sqrt(file_mse)
        file_mae = mean_absolute_error(y_actual_file, y_pred_file)
        file_r2 = r2_score(y_actual_file, y_pred_file)

        per_file_metrics[filename] = {
            'MSE': float(file_mse),
            'RMSE': float(file_rmse),
            'MAE': float(file_mae),
            'R2': float(file_r2)
        }

        print(f"  R²: {file_r2:.4f}, RMSE: {file_rmse:.4f}, MAE: {file_mae:.4f}")

        plot_path = plot_predictions_per_file(filename, y_pred_file, y_actual_file, EVAL_RESULT_FOLDER)
        print(f"  ✅ Plot: {os.path.basename(plot_path)}")

        all_y_actual_files.extend(y_actual_file)
        all_y_pred_files.extend(y_pred_file)

    # COMBINED RESIDUAL PLOT
    if len(all_y_actual_files) > 0:
        print(f"\n📊 Creating combined residual analysis...")
        all_y_actual_files = np.array(all_y_actual_files)
        all_y_pred_files = np.array(all_y_pred_files)

        combined_plot_path = plot_combined_residuals(all_y_actual_files, all_y_pred_files, EVAL_RESULT_FOLDER)
        print(f"✅ Combined plot: {os.path.basename(combined_plot_path)}")

        all_y_actual_files = clean_array(all_y_actual_files, "combined actual")
        all_y_pred_files = clean_array(all_y_pred_files, "combined predicted")

        combined_mse = mean_squared_error(all_y_actual_files, all_y_pred_files)
        combined_rmse = np.sqrt(combined_mse)
        combined_mae = mean_absolute_error(all_y_actual_files, all_y_pred_files)
        combined_r2 = r2_score(all_y_actual_files, all_y_pred_files)

        print(f"\n📈 Combined Metrics:")
        print(f"  MSE:  {combined_mse:.6f}")
        print(f"  RMSE: {combined_rmse:.4f} nm")
        print(f"  MAE:  {combined_mae:.4f} nm")
        print(f"  R²:   {combined_r2:.4f}")
    else:
        combined_mse = combined_rmse = combined_mae = combined_r2 = 0.0
        print(f"\n⚠️ No per-file predictions — skipped combined residual analysis")

    # SAVE REPORTS
    print(f"\n📝 Saving evaluation reports...")

    report = {
        'evaluation_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'method1_exact_training_evaluation': {
            'mse': float(mse), 'rmse': float(rmse), 'mae': float(mae), 'r2': float(r2)
        },
        'method2_per_file_combined': {
            'mse': float(combined_mse), 'rmse': float(combined_rmse),
            'mae': float(combined_mae), 'r2': float(combined_r2)
        },
        'per_file_metrics': per_file_metrics,
        'config': {
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'loss_type': args.loss_type,
            'use_kalman_filter': USE_KALMAN_FILTER,
            'input_features': INPUT_FEATURES
        }
    }

    with open(os.path.join(EVAL_RESULT_FOLDER, 'comprehensive_metrics.json'), 'w') as f:
        json.dump(report, f, indent=2)

    df = pd.DataFrame([{'Filename': k, **v} for k, v in per_file_metrics.items()])
    df.to_csv(os.path.join(EVAL_RESULT_FOLDER, 'per_file_metrics.csv'), index=False)

    print(f"\n{'='*70}")
    print(f" EVALUATION COMPLETE!")
    print(f"{'='*70}")

    return {
        'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2,
        'combined_r2': combined_r2, 'combined_rmse': combined_rmse
    }

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
            'ATTENTION_ITERATIONS': ATTENTION_ITERATIONS,
            'BATCH_SIZE': BATCH_SIZE,
            'LEARNING_RATE': LEARNING_RATE,
            'EARLY_STOP_PATIENCE': EARLY_STOP_PATIENCE,
            'SEQUENCE_STEP': SEQUENCE_STEP,
            'USE_MIXED_PRECISION': USE_MIXED_PRECISION,
            'USE_KALMAN_FILTER': USE_KALMAN_FILTER,
            'INPUT_FEATURES': INPUT_FEATURES
        }
    }
    if not args.deepspeed or args.local_rank == 0:
        torch.save(checkpoint, filename)
        print(f"✅ Checkpoint saved: {filename}")

def load_checkpoint(filename, model, optimizer, scheduler, device):
    """Load checkpoint with proper handling for 'module.' prefix"""
    if os.path.isfile(filename):
        print(f"🔄 Loading checkpoint: {filename}")
        checkpoint = torch.load(filename, map_location=device, weights_only=False)

        model_state_dict = checkpoint['model_state_dict']
        print(f"  Checkpoint state dict keys (first 3): {list(model_state_dict.keys())[:3]}")

        if hasattr(model, 'module'):
            print(f"  Model is DeepSpeed wrapped")
            has_module_prefix = any(key.startswith('module.') for key in model_state_dict.keys())

            if has_module_prefix:
                print(f"  Stripping 'module.' prefix from checkpoint...")
                clean_state_dict = {}
                for key, value in model_state_dict.items():
                    new_key = key.replace('module.', '', 1) if key.startswith('module.') else key
                    clean_state_dict[new_key] = value
                model.module.load_state_dict(clean_state_dict)
                print(f"  ✅ Model state loaded (stripped prefix)")
            else:
                model.module.load_state_dict(model_state_dict)
                print(f"  ✅ Model state loaded directly")
        else:
            print(f"  Model is NOT DeepSpeed wrapped")
            has_module_prefix = any(key.startswith('module.') for key in model_state_dict.keys())

            if has_module_prefix:
                print(f"  Stripping 'module.' prefix from checkpoint...")
                clean_state_dict = {}
                for key, value in model_state_dict.items():
                    new_key = key.replace('module.', '', 1) if key.startswith('module.') else key
                    clean_state_dict[new_key] = value
                model.load_state_dict(clean_state_dict)
                print(f"  ✅ Model state loaded (stripped prefix)")
            else:
                model.load_state_dict(model_state_dict)
                print(f"  ✅ Model state loaded directly")

        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("  ✅ Optimizer state loaded")
        except (KeyError, Exception) as e:
            print(f"  ⚠️ Could not load optimizer state: {e} — skipping")

        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("  ✅ Scheduler state loaded")
        except (KeyError, Exception) as e:
            print(f"  ⚠️ Could not load scheduler state: {e} — skipping")

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
        print(f"⚠️ No checkpoint found: {filename}")
        return 0, [], [], [], float('inf'), 0, time.time(), None, None

# ============================================================================
# METRICS LOGGING FUNCTIONS
# ============================================================================
def save_metrics_to_csv(epoch_metrics, filename):
    df = pd.DataFrame(epoch_metrics)
    df.to_csv(filename, index=False)
    print(f"✅ Metrics saved: {filename}")

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
# MAIN EXECUTION - MEMORY EFFICIENT
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepspeed", action="store_true", help="Enable DeepSpeed")
    parser.add_argument("--deepspeed_config", type=str, help="DeepSpeed config file")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    parser.add_argument("--loss_type", type=str, default="mse",
                       choices=["mse", "huber", "huber_gradient", "adaptive", "peak_aware"],
                       help="Type of loss function to use (default: mse — matches base script)")
    parser.add_argument("--skip_evaluation", action="store_true",
                       help="Skip the integrated evaluation at the end")
    args = parser.parse_args()

    print("\n" + "="*70)
    print(f"  xLSTM VIBRATION PREDICTION - {NUM_LAYERS}-LAYER STACK WITH KALMAN FILTER")
    print(f"  Loss Function: {args.loss_type}")
    print(f"  Kalman Filter: {'Enabled' if USE_KALMAN_FILTER else 'Disabled'}")
    print(f"  Input Features: {INPUT_FEATURES} (x, v, a, jerk, causal peak flag)")
    print("="*70)
    print(f"\nFolders:")
    print(f"  Train Folder:    {TRAIN_FOLDER}")
    print(f"  Val Folder:      {VAL_FOLDER}")
    print(f"  Test Folder:     {TEST_FOLDER} (final evaluation only)")
    print(f"  Sequence Length: {SEQUENCE_LENGTH}")
    print(f"  Stacking:        {NUM_LAYERS} mLSTM + {NUM_LAYERS} sLSTM")
    print(f"  Batch Size:      {BATCH_SIZE}")
    print(f"  Learning Rate:   {LEARNING_RATE}")
    print(f"  Max Epochs:      {NUM_EPOCHS}")
    print(f"  Precision:       float32")
    print("="*70)

    os.makedirs(RESULT_FOLDER, exist_ok=True)
    os.makedirs(EVAL_RESULT_FOLDER, exist_ok=True)

    # Prepare datasets with Kalman filter
    X_train, X_val, y_train, y_val, train_series, val_series = prepare_datasets_with_kf(
        train_folder=TRAIN_FOLDER,
        val_folder=VAL_FOLDER,
        sequence_length=SEQUENCE_LENGTH,
        step=SEQUENCE_STEP,
        use_kf=USE_KALMAN_FILTER
    )

    # No-lookahead check on the FEATURE BUILDER, before any training: this is
    # the check the legacy jerk estimator failed (FIX 6 in the header).
    if USE_KALMAN_FILTER:
        assert_kf_features_causal(train_series[0], lookahead=PEAK_LOOKAHEAD)
        assert_kf_features_causal(val_series[0], lookahead=PEAK_LOOKAHEAD)

    # Scale data in batches (IN-PLACE to save memory)
    print(f"\nScaling data...")

    # Fit scalers on a sample to avoid memory issues.
    # FIX 9: draw the sample UNIFORMLY AT RANDOM over the whole training set.
    # X_train[:sample_size] is not a sample of the corpus: the windows are
    # concatenated file by file, so when the alphabetically first file holds
    # more than SCALER_SAMPLE_SIZE windows (Lathe: 316,694 of 5,552,151;
    # synthetic: 63,900 of 1,533,600) every scaler statistic came from that
    # ONE recording. On Lathe that file is the quietest in the set (std 439
    # against 3.1e4 for the loudest), so the other 16 files were scaled to
    # tens of sigma, nearly every shuffled batch tripped the `loss > 100`
    # guard, and the run trained on nothing. Seeded, so all ranks agree.
    sample_size = min(SCALER_SAMPLE_SIZE, len(X_train))
    if sample_size >= len(X_train):
        sample_idx = np.arange(len(X_train))
    else:
        sample_idx = np.random.default_rng(SCALER_SEED).choice(
            len(X_train), size=sample_size, replace=False)
    sample_X = X_train[sample_idx]
    sample_y = y_train[sample_idx]

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    sample_X_flat = sample_X.reshape(sample_size, -1)
    feature_scaler.fit(sample_X_flat.astype(np.float32))
    target_scaler.fit(sample_y.reshape(-1, 1).astype(np.float32))

    draw_kind = "all" if sample_size >= len(X_train) else "random draw"
    print(f"✅ Scalers fit on {sample_size:,} samples ({draw_kind}) of "
          f"{len(X_train):,} training windows, seed={SCALER_SEED}")
    print(f"   target scaler: mean={target_scaler.mean_[0]:.6g}  "
          f"scale={target_scaler.scale_[0]:.6g}")
    del sample_X, sample_y, sample_X_flat, sample_idx

    # Scale all data using memory-efficient function (IN-PLACE)
    X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled = scale_data_memory_efficient(
        X_train, X_val, y_train, y_val, feature_scaler, target_scaler, use_kf=USE_KALMAN_FILTER
    )

    print("✅ All data scaled successfully!")

    # FIX 2 (prep): compute the peak threshold from the FULL training-target
    # distribution while the numpy array still exists. Previously the threshold
    # was cached from the first arbitrary training batch — and under multi-GPU
    # DeepSpeed each rank computed its own different value. Applied to the
    # criterion after it is created below.
    # FIX 9 (diagnostic): with a scaler that represents the corpus, the
    # standardised training targets sit within a few sigma. A range of tens of
    # sigma means some files are far outside the fit and the `loss > 100`
    # guard at the top of the training loop will start dropping their batches.
    print(f"   scaled train targets: [{y_train_scaled.min():.3f}, "
          f"{y_train_scaled.max():.3f}]  (well-scaled data stays within a few sigma)")
    peak_threshold_from_train = float(np.quantile(y_train_scaled, PEAK_THRESHOLD_QUANTILE))

    # Free original data (we no longer need them)
    del X_train, X_val, y_train, y_val

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)

    # Free the scaled numpy arrays (we have tensors now)
    del X_train_scaled, X_val_scaled, y_train_scaled, y_val_scaled

    # Build datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    # Initialize model with multi-channel input
    model = ImprovedxLSTMPredictor_MultiChannel(
        input_size=INPUT_FEATURES if USE_KALMAN_FILTER else 1,
        hidden_size=HIDDEN_SIZE,
        mem_dim=MEMORY_DIM,
        output_size=1,
        num_layers=NUM_LAYERS,
        attention_iterations=ATTENTION_ITERATIONS
    )

    # Initialize loss function (default: plain MSE — FIX 6, matches base script)
    if args.loss_type == "mse":
        criterion = nn.MSELoss()
    elif args.loss_type == "huber":
        criterion = nn.HuberLoss(delta=1.0)
    elif args.loss_type == "huber_gradient":
        criterion = HuberGradientLoss(
            huber_delta=1.0, alpha=1.0, beta=0.05, gamma=0.05,
            underpred_penalty=1.2, gradient_threshold=0.2
        )
    elif args.loss_type == "adaptive":
        criterion = AdaptiveHuberGradientLoss(
            huber_delta=1.0, alpha_init=1.0, beta_init=0.1, gamma_init=0.1,
            underpred_penalty=2.0, gradient_threshold=0.5, total_epochs=NUM_EPOCHS
        )
    elif args.loss_type == "peak_aware":
        criterion = PeakAwareLoss(
            huber_delta=1.0, alpha=1.0, beta=0.3, gamma=0.2,
            peak_penalty=10.0, peak_threshold_quantile=PEAK_THRESHOLD_QUANTILE,
            underpred_penalty=4.0, gradient_threshold=0.5
        )

    # FIX 2 (apply): deterministic, rank-consistent peak threshold
    if hasattr(criterion, 'peak_threshold'):
        criterion.peak_threshold = peak_threshold_from_train
        print(f"✅ Peak threshold set from full training targets: {criterion.peak_threshold:.6f}")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # DeepSpeed initialization
    if args.deepspeed and args.deepspeed_config:
        print("🚀 Initializing DeepSpeed...")
        model, optimizer, _, _ = deepspeed.initialize(
            args=args, model=model, optimizer=optimizer
        )
        device = model.local_rank
        print(f"✅ DeepSpeed initialized! Local rank: {device}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        if hasattr(criterion, 'to'):
            criterion = criterion.to(device)

    # DataLoaders
    if args.deepspeed:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
        shuffle=(train_sampler is None), num_workers=0, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler,
        shuffle=False, num_workers=0, pin_memory=True
    )

    print(f"\n{'='*70}")
    print(f"  STARTING TRAINING")
    print(f"{'='*70}")
    print(f"Training batches per GPU: {len(train_loader)}")
    print(f"Validation batches per GPU: {len(val_loader)}")
    print(f"{'='*70}\n")

    # Checkpoint loading
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
    else:
        print("🚀 Starting fresh training")
        training_start_time = time.time()
        if not args.deepspeed or args.local_rank == 0:
            if os.path.exists(VAL_LOSS_FILE):
                os.remove(VAL_LOSS_FILE)
            if os.path.exists(VAL_ACCURACY_FILE):
                os.remove(VAL_ACCURACY_FILE)

    epoch_metrics = []
    if start_epoch > 0:
        for i in range(len(train_losses)):
            epoch_metrics.append({
                'epoch': i + 1,
                'train_loss': round(train_losses[i], 6),
                'val_loss': round(val_losses[i], 6),
                'val_r2_score': round(val_r2_scores[i], 4) if i < len(val_r2_scores) else 0.0,
                'epoch_time_sec': 0,
                'learning_rate': LEARNING_RATE,
                'loss_huber': 0.0, 'loss_gradient': 0.0, 'loss_asymmetric': 0.0, 'loss_peak': 0.0
            })

    # Training loop
    for epoch in range(start_epoch, NUM_EPOCHS):
        epoch_start_time = time.time()

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if args.loss_type == "adaptive":
            criterion.update_weights(epoch)

        model.train()
        epoch_loss = 0
        valid_batches = 0
        loss_components_sum = {'huber': 0.0, 'gradient': 0.0, 'asymmetric': 0.0, 'peak': 0.0}

        progress_bar = tqdm(
            train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}',
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
                optimizer.step()

            epoch_loss += loss.item()
            valid_batches += 1

            if hasattr(criterion, 'last_components'):
                for key in loss_components_sum:
                    if key in criterion.last_components:
                        loss_components_sum[key] += criterion.last_components[key]

            if args.loss_type in ["huber_gradient", "adaptive", "peak_aware"] and hasattr(criterion, 'last_components'):
                postfix = {'Loss': f'{loss.item():.4f}'}
                if 'huber' in criterion.last_components:
                    postfix['Huber'] = f'{criterion.last_components.get("huber", 0):.4f}'
                if 'gradient' in criterion.last_components:
                    postfix['Grad'] = f'{criterion.last_components.get("gradient", 0):.4f}'
                if 'asymmetric' in criterion.last_components:
                    postfix['Asym'] = f'{criterion.last_components.get("asymmetric", 0):.4f}'
                if 'peak' in criterion.last_components:
                    postfix['Peak'] = f'{criterion.last_components.get("peak", 0):.4f}'
                progress_bar.set_postfix(postfix)
            else:
                progress_bar.set_postfix({'Loss': f'{loss.item():.6f}'})

        train_loss = epoch_loss / max(valid_batches, 1)
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

        # Cast to float64 before inverse_transform (parity with the base
        # script — guarantees the metrics pipeline is dtype-safe)
        val_predictions = target_scaler.inverse_transform(
            np.array(val_predictions, dtype=np.float64).reshape(-1, 1)).flatten()
        val_actuals = target_scaler.inverse_transform(
            np.array(val_actuals, dtype=np.float64).reshape(-1, 1)).flatten()
        val_r2 = calculate_r2_score(val_actuals, val_predictions)

        epoch_time = time.time() - epoch_start_time

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_r2_scores.append(val_r2)

        avg_components = {}
        if valid_batches > 0:
            for key in loss_components_sum:
                avg_components[key] = loss_components_sum[key] / valid_batches

        epoch_metrics.append({
            'epoch': epoch + 1,
            'train_loss': round(train_loss, 6),
            'val_loss': round(val_loss, 6),
            'val_r2_score': round(val_r2, 4),
            'epoch_time_sec': round(epoch_time, 2),
            'learning_rate': optimizer.param_groups[0]['lr'],
            'loss_huber': round(avg_components.get('huber', 0), 6),
            'loss_gradient': round(avg_components.get('gradient', 0), 6),
            'loss_asymmetric': round(avg_components.get('asymmetric', 0), 6),
            'loss_peak': round(avg_components.get('peak', 0), 6)
        })

        if not args.deepspeed or args.local_rank == 0:
            save_metrics_to_csv(epoch_metrics, METRICS_FILE)
            save_history_to_txt(epoch + 1, val_loss, val_r2, VAL_LOSS_FILE, VAL_ACCURACY_FILE, args)

            print(f"\n{'='*70}")
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Time: {epoch_time:.1f}s")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss:   {val_loss:.6f}")
            print(f"  Val R²:     {val_r2:.4f}")
            print(f"  LR:         {optimizer.param_groups[0]['lr']:.6f}")

            if args.loss_type in ["huber_gradient", "adaptive", "peak_aware"] and avg_components:
                print(f"  Loss Components:")
                print(f"    Huber:     {avg_components.get('huber', 0):.6f}")
                print(f"    Gradient:  {avg_components.get('gradient', 0):.6f}")
                print(f"    Asymmetric: {avg_components.get('asymmetric', 0):.6f}")
                if 'peak' in avg_components:
                    print(f"    Peak Penalty: {avg_components.get('peak', 0):.6f}")

                if args.loss_type == "adaptive" and hasattr(criterion, 'base_loss'):
                    print(f"  Adaptive Weights:")
                    print(f"    α={criterion.base_loss.alpha:.3f}, β={criterion.base_loss.beta:.3f}, γ={criterion.base_loss.gamma:.3f}")

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
            epoch=epoch, model=model, optimizer=optimizer, scheduler=scheduler,
            train_losses=train_losses, val_losses=val_losses, val_r2_scores=val_r2_scores,
            best_val_loss=best_val_loss, patience_counter=patience_counter,
            training_start_time=training_start_time, filename=CHECKPOINT_FILE,
            args=args, feature_scaler=feature_scaler, target_scaler=target_scaler
        )

        if patience_counter >= EARLY_STOP_PATIENCE:
            if not args.deepspeed or args.local_rank == 0:
                print(f"\n🛑 Early stopping triggered at epoch {epoch+1}")
            break

    training_time = time.time() - training_start_time

    # Final evaluation and model saving (only main process)
    if not args.deepspeed or args.local_rank == 0:
        print(f"\n{'='*70}")
        print(f"  TRAINING COMPLETED")
        print(f"{'='*70}")
        print(f"Total training time: {training_time/3600:.2f} hours ({training_time/60:.1f} minutes)")
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"Best validation R²: {max(val_r2_scores):.4f}")
        print(f"{'='*70}\n")

        # Load best model for saving
        model_path = os.path.join(RESULT_FOLDER, 'best_xlstm_model.pth')
        if os.path.exists(model_path):
            print(f"🔄 Loading best model from: {model_path}")

            best_model_state = torch.load(model_path, map_location=map_device, weights_only=False)

            if hasattr(model, 'module'):
                has_module_prefix = any(key.startswith('module.') for key in best_model_state.keys())

                if has_module_prefix:
                    clean_state_dict = {}
                    for key, value in best_model_state.items():
                        new_key = key.replace('module.', '', 1) if key.startswith('module.') else key
                        clean_state_dict[new_key] = value
                    model.module.load_state_dict(clean_state_dict)
                else:
                    model.module.load_state_dict(best_model_state)
            else:
                has_module_prefix = any(key.startswith('module.') for key in best_model_state.keys())

                if has_module_prefix:
                    clean_state_dict = {}
                    for key, value in best_model_state.items():
                        new_key = key.replace('module.', '', 1) if key.startswith('module.') else key
                        clean_state_dict[new_key] = value
                    model.load_state_dict(clean_state_dict)
                else:
                    model.load_state_dict(best_model_state)

            print("✅ Best model loaded for final save\n")

        # Save final model with all metadata
        save_dict = {
            'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'feature_scaler': feature_scaler,
            'target_scaler': target_scaler,
            'sequence_length': SEQUENCE_LENGTH,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_r2_scores': val_r2_scores,
            'best_val_loss': best_val_loss,
            'training_time_hours': training_time / 3600,
            'loss_type': args.loss_type,
            'use_kalman_filter': USE_KALMAN_FILTER,
            'metrics': {
                'mse': 0, 'rmse': 0, 'mae': 0, 'r2': max(val_r2_scores)
            },
            'config': {
                'hidden_size': HIDDEN_SIZE,
                'memory_dim': MEMORY_DIM,
                'num_layers': NUM_LAYERS,
                'batch_size': BATCH_SIZE,
                'learning_rate': LEARNING_RATE,
                'mixed_precision': USE_MIXED_PRECISION,
                'loss_type': args.loss_type,
                'input_features': INPUT_FEATURES,
                'use_kalman_filter': USE_KALMAN_FILTER
            }
        }
        final_model_path = os.path.join(RESULT_FOLDER, 'xlstm_vibration_predictor_final.pth')
        torch.save(save_dict, final_model_path)
        print(f"✅ Model saved: {final_model_path}")

        # Save training plots
        fig = plt.figure(figsize=(16, 10))

        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(train_losses, label='Training Loss', linewidth=2)
        ax1.plot(val_losses, label='Validation Loss', linewidth=2)
        ax1.set_title('Training History', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
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
        if args.loss_type in ["huber_gradient", "adaptive", "peak_aware"]:
            epochs = range(1, len(epoch_metrics) + 1)
            huber_vals = [m.get('loss_huber', 0) for m in epoch_metrics]
            grad_vals = [m.get('loss_gradient', 0) for m in epoch_metrics]
            asym_vals = [m.get('loss_asymmetric', 0) for m in epoch_metrics]
            peak_vals = [m.get('loss_peak', 0) for m in epoch_metrics]

            if any(v > 0 for v in huber_vals):
                ax3.plot(epochs, huber_vals, label='Huber Loss', linewidth=1.5)
                ax3.plot(epochs, grad_vals, label='Gradient Loss', linewidth=1.5)
                ax3.plot(epochs, asym_vals, label='Asymmetric Loss', linewidth=1.5)
                if any(v > 0 for v in peak_vals):
                    ax3.plot(epochs, peak_vals, label='Peak Penalty', linewidth=1.5)
                ax3.set_title('Loss Components', fontsize=12, fontweight='bold')
                ax3.set_xlabel('Epoch')
                ax3.set_ylabel('Loss Value')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            else:
                ax3.axis('off')
        else:
            ax3.axis('off')

        ax4 = plt.subplot(2, 3, 4)
        ax4.axis('off')
        info_text = f"""
        TRAINING SUMMARY
        Total Time: {training_time/3600:.2f}h
        Best Val Loss: {best_val_loss:.6f}
        Best R²: {max(val_r2_scores):.4f}
        Epochs: {len(train_losses)}

        Model: {NUM_LAYERS}-Layer xLSTM
        Hidden: {HIDDEN_SIZE}, Mem: {MEMORY_DIM}
        Batch: {BATCH_SIZE}, LR: {LEARNING_RATE}
        Loss: {args.loss_type}
        Precision: float32
        Kalman Filter: {'ON' if USE_KALMAN_FILTER else 'OFF'}
        Input Features: {INPUT_FEATURES} (x, v, a, jerk, causal peak flag)
        """
        ax4.text(0.1, 0.5, info_text, fontsize=10, family='monospace',
                verticalalignment='center')

        plt.tight_layout()
        plot_path = os.path.join(RESULT_FOLDER, 'xlstm_training_summary.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Training summary plot saved: {plot_path}")

        # ====================================================================
        # INTEGRATED EVALUATION (runs automatically after training)
        # ====================================================================
        if not args.skip_evaluation:
            print(f"\n{'='*70}")
            print(f"  STARTING INTEGRATED EVALUATION")
            print(f"{'='*70}")

            if hasattr(model, 'module'):
                eval_model = model.module
            else:
                eval_model = model

            eval_model.eval()

            os.makedirs(EVAL_RESULT_FOLDER, exist_ok=True)
            print(f"Evaluation results folder: {EVAL_RESULT_FOLDER}")

            try:
                eval_metrics = run_full_evaluation(
                    model=eval_model,
                    feature_scaler=feature_scaler,
                    target_scaler=target_scaler,
                    device=device,
                    args=args
                )

                if eval_metrics:
                    save_dict['metrics'] = {
                        'mse': float(eval_metrics['mse']),
                        'rmse': float(eval_metrics['rmse']),
                        'mae': float(eval_metrics['mae']),
                        'r2': float(eval_metrics['r2']),
                        'combined_r2': float(eval_metrics['combined_r2']),
                        'combined_rmse': float(eval_metrics['combined_rmse'])
                    }
                    torch.save(save_dict, final_model_path)
                    print(f"✅ Updated final model with evaluation metrics")
            except Exception as e:
                print(f"❌ Evaluation failed with error: {e}")
                import traceback
                traceback.print_exc()

        print("\n" + "="*70)
        print("  TRAINING AND EVALUATION COMPLETE!")
        print("="*70)
        print(f"\n📁 All results saved to: {RESULT_FOLDER}")
        print(f"   - xlstm_vibration_predictor_final.pth (final model with all metadata)")
        print(f"   - best_xlstm_model.pth (best model during training)")
        print(f"   - xlstm_training_summary.png (training visualization)")
        print(f"   - training_metrics.csv (epoch-by-epoch metrics)")
        print(f"   - val_loss_history.txt (loss history)")
        print(f"   - val_accuracy_history.txt (accuracy history)")
        print(f"   - xlstm_deepspeed_checkpoint.pth (checkpoint for resume)")
        if not args.skip_evaluation:
            print(f"\n📁 Evaluation results: {EVAL_RESULT_FOLDER}")
            print(f"   - predictions_*.png (individual file plots)")
            print(f"   - combined_residual_analysis.png")
            print(f"   - comprehensive_metrics.json")
            print(f"   - per_file_metrics.csv")
        print("="*70)