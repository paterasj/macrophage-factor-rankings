from google.colab import drive
drive.flush_and_unmount()
drive.mount('/content/drive', force_remount=True)

# -*- coding: utf-8 -*-

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


# =========================================================
# GLOBAL PLOT SETTINGS
# =========================================================
def set_plot_text(scale=1.5):
    import matplotlib as mpl
    base = 10 * scale
    mpl.rcParams.update({
        "font.size": base,
        "axes.titlesize": base * 1.15,
        "axes.labelsize": base * 1.05,
        "xtick.labelsize": base * 0.95,
        "ytick.labelsize": base * 0.95,
        "legend.fontsize": base * 0.9,
        "figure.titlesize": base * 1.2,
        "lines.linewidth": 1.8,
        "axes.linewidth": 1.1,
    })


set_plot_text(scale=1.35)


# =========================================================
# CONFIG
# =========================================================
DATA_PATH_1 = "/content/drive/MyDrive/macrophage_M1_Scan"
DATA_PATH_2 = "/content/drive/MyDrive/macrophage_M2_Scan"


TIME_COLUMN = "Time"
TRAJECTORY_ID_COLUMN = None   # Set to a column name if your data has an explicit sample/trajectory id

BATCH_SIZE = 32
TEST_SIZE = 0.30
RANDOM_STATE = 42
LEARNING_RATE = 1e-4
EPOCHS = 10

LSTM_HIDDEN_DIM = 128
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.20

SAVE_SHAP_PLOTS = True
SAVE_ICE_PDP_PLOTS = False
SHAP_PLOT_DIR = "/content/drive/MyDrive/SHAP_plots"
ICE_PDP_DIR = "Plot/ICE_PDP_plots_LSTM"

CYTOKINE_CHEMOKINE_FEATURES = [
    "[IFNG]", "[CXCL9]", "[mCXCL10]", "[iNOS]",
    "[TNFa]", "[IL12]", "[VEGF]", "[IL4]", "[IL10]", "[ARG1]"
]

TF_FEATURES = [
    "[STAT1]", "[IRF9]", "[STAT6]", "[PPARg]", "[IRF4]", "[IRF1]"
]

TARGET_COLUMNS = [
    "[IFNG]", "[iNOS]", "[TNFa]", "[IL12]", "[mCXCL10]", "[CXCL9]",
    "[STAT1]", "[ARG1]", "[IL4]", "[VEGF]", "[IL10]", "[STAT6]", "[IRF9]", "[PPARG]", "[IRF4]", "[IRF1]"
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# =========================================================
# REPRODUCIBILITY
# =========================================================
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(RANDOM_STATE)


# =========================================================
# DATA LOADING
# =========================================================
df1 = pd.read_csv(DATA_PATH_1, sep=",")
df2 = pd.read_csv(DATA_PATH_2, sep=",")
df = pd.concat([df1, df2], ignore_index=True)
df = df.apply(pd.to_numeric, errors="coerce")

print(f"Shape of df1: {df1.shape}")
print(f"Shape of df2: {df2.shape}")
print(f"Shape of merged df: {df.shape}")

if TIME_COLUMN not in df.columns:
    raise ValueError(f"Required time column '{TIME_COLUMN}' not found in dataframe.")

df = df.dropna(subset=[TIME_COLUMN]).copy()


# =========================================================
# HELPERS: BUILD FULL TRAJECTORIES
# =========================================================
def build_trajectories(df, time_col="Time", id_col=None):
    """
    Build full trajectories from a long dataframe.

    If id_col is provided, rows are grouped by that identifier and sorted by time.
    Otherwise, this assumes that the row order within each time block corresponds
    to the same sample/trajectory across timepoints, which matches the original
    static initial/final alignment logic.
    """
    df = df.copy()
    unique_times = np.sort(df[time_col].unique())
    feature_cols = [c for c in df.columns if c != time_col and c != id_col]

    if id_col is not None and id_col in df.columns:
        grouped = []
        kept_ids = []

        for traj_id, g in df.groupby(id_col):
            g_sorted = g.sort_values(time_col)
            times_here = g_sorted[time_col].to_numpy()

            if len(times_here) != len(unique_times):
                continue
            if not np.array_equal(times_here, unique_times):
                continue

            grouped.append(g_sorted[feature_cols].to_numpy(dtype=np.float32))
            kept_ids.append(traj_id)

        if len(grouped) == 0:
            raise ValueError("No complete trajectories found using the supplied trajectory id column.")

        X_all = np.stack(grouped, axis=0)  # (n_samples, n_times, n_features)
        return X_all, unique_times, feature_cols, kept_ids

    # Fallback: reconstruct trajectories by row order within each timepoint
    blocks = []
    counts = []

    for t in unique_times:
        block = df[df[time_col] == t][feature_cols].reset_index(drop=True)
        blocks.append(block)
        counts.append(len(block))

    if len(set(counts)) != 1:
        raise ValueError(
            "Cannot reconstruct trajectories by row order because timepoints have different row counts. "
            "Provide an explicit trajectory/sample id column via TRAJECTORY_ID_COLUMN."
        )

    n_samples = counts[0]
    n_times = len(unique_times)
    n_features = len(feature_cols)

    X_all = np.zeros((n_samples, n_times, n_features), dtype=np.float32)
    for ti, block in enumerate(blocks):
        X_all[:, ti, :] = block.to_numpy(dtype=np.float32)

    return X_all, unique_times, feature_cols, None


all_sequences, unique_times, feature_names, trajectory_ids = build_trajectories(
    df=df,
    time_col=TIME_COLUMN,
    id_col=TRAJECTORY_ID_COLUMN
)

print(f"\nNumber of trajectories: {all_sequences.shape[0]}")
print(f"Number of timepoints:   {all_sequences.shape[1]}")
print(f"Number of features:     {all_sequences.shape[2]}")
print(f"Times: {unique_times.tolist()}")


# =========================================================
# INPUT / TARGET PREPARATION FOR SEQUENCE LEARNING
# =========================================================
target_indices = [feature_names.index(col) for col in TARGET_COLUMNS]
target_names = TARGET_COLUMNS.copy()

# Many-to-many next-step forecasting:
# input at steps 0..T-2  -> target at steps 1..T-1
X_seq_raw = all_sequences[:, :-1, :]                          # (N, T-1, F_in)
y_seq_raw = all_sequences[:, 1:, :][:, :, target_indices]    # (N, T-1, F_out)

seq_len = X_seq_raw.shape[1]
input_dim = X_seq_raw.shape[2]
output_dim = y_seq_raw.shape[2]

print(f"\nInput sequence shape:  {X_seq_raw.shape}")
print(f"Target sequence shape: {y_seq_raw.shape}")
print(f"Sequence length used for training: {seq_len}")


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================
train_idx, test_idx = train_test_split(
    np.arange(X_seq_raw.shape[0]),
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    shuffle=True
)

X_train_raw = X_seq_raw[train_idx]
X_test_raw = X_seq_raw[test_idx]
y_train_raw = y_seq_raw[train_idx]
y_test_raw = y_seq_raw[test_idx]


# =========================================================
# SCALING
# =========================================================
def fit_input_scaler_3d(X_train_3d):
    """
    Fit a single scaler across all timepoints for each input feature.
    """
    n_train, t_len, n_feat = X_train_3d.shape
    scaler = StandardScaler()
    scaler.fit(X_train_3d.reshape(-1, n_feat))
    return scaler


def transform_3d_with_scaler(X_3d, scaler):
    n, t, f = X_3d.shape
    X_scaled = scaler.transform(X_3d.reshape(-1, f)).reshape(n, t, f)
    return X_scaled.astype(np.float32)


def fit_output_scalers_3d(y_train_3d, target_names):
    """
    Fit one scaler per output target across all timepoints.
    """
    output_scalers = {}
    scaled = np.zeros_like(y_train_3d, dtype=np.float32)

    for j, name in enumerate(target_names):
        scaler = StandardScaler()
        col = y_train_3d[:, :, j].reshape(-1, 1)
        scaler.fit(col)
        scaled[:, :, j] = scaler.transform(col).reshape(y_train_3d.shape[0], y_train_3d.shape[1])
        output_scalers[name] = scaler

    return scaled, output_scalers


def transform_outputs_3d(y_3d, output_scalers, target_names):
    scaled = np.zeros_like(y_3d, dtype=np.float32)
    for j, name in enumerate(target_names):
        scaler = output_scalers[name]
        col = y_3d[:, :, j].reshape(-1, 1)
        scaled[:, :, j] = scaler.transform(col).reshape(y_3d.shape[0], y_3d.shape[1])
    return scaled.astype(np.float32)


def inverse_transform_outputs_3d(y_scaled_3d, output_scalers, target_names):
    y_real = np.zeros_like(y_scaled_3d, dtype=np.float32)
    for j, name in enumerate(target_names):
        scaler = output_scalers[name]
        col = y_scaled_3d[:, :, j].reshape(-1, 1)
        y_real[:, :, j] = scaler.inverse_transform(col).reshape(y_scaled_3d.shape[0], y_scaled_3d.shape[1])
    return y_real


input_scaler = fit_input_scaler_3d(X_train_raw)
X_train = transform_3d_with_scaler(X_train_raw, input_scaler)
X_test = transform_3d_with_scaler(X_test_raw, input_scaler)

y_train, output_scalers = fit_output_scalers_3d(y_train_raw, target_names)
y_test = transform_outputs_3d(y_test_raw, output_scalers, target_names)

print("\nScaled train/test arrays ready.")


# =========================================================
# DATASET / DATALOADER
# =========================================================
class MacrophageSequenceDataset(Dataset):
    def __init__(self, X_seq, y_seq):
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.y_seq = torch.tensor(y_seq, dtype=torch.float32)

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        return self.X_seq[idx], self.y_seq[idx]


train_dataset = MacrophageSequenceDataset(X_train, y_train)
test_dataset = MacrophageSequenceDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# =========================================================
# MODEL
# =========================================================
class LSTMTimeSeriesModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.2):
        super().__init__()

        effective_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=effective_dropout,
            batch_first=True
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)           # (batch, seq_len, hidden_dim)
        preds = self.head(lstm_out)          # (batch, seq_len, output_dim)
        return preds


model = LSTMTimeSeriesModel(
    input_dim=input_dim,
    hidden_dim=LSTM_HIDDEN_DIM,
    output_dim=output_dim,
    num_layers=LSTM_NUM_LAYERS,
    dropout=LSTM_DROPOUT
).to(DEVICE)

loss_fn = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# =========================================================
# TRAINING / EVALUATION HELPERS
# =========================================================
def compute_sequence_metrics(y_true_scaled, y_pred_scaled, target_names):
    """
    Metrics in scaled space for optimization tracking.
    """
    n, t, d = y_true_scaled.shape
    y_true_flat = y_true_scaled.reshape(-1, d)
    y_pred_flat = y_pred_scaled.reshape(-1, d)

    per_target_r2_all_steps = r2_score(y_true_flat, y_pred_flat, multioutput="raw_values")
    overall_r2_all_steps = r2_score(y_true_flat, y_pred_flat, multioutput="uniform_average")

    y_true_final = y_true_scaled[:, -1, :]
    y_pred_final = y_pred_scaled[:, -1, :]
    per_target_r2_final = r2_score(y_true_final, y_pred_final, multioutput="raw_values")
    overall_r2_final = r2_score(y_true_final, y_pred_final, multioutput="uniform_average")

    metrics = {
        "overall_r2_all_steps": overall_r2_all_steps,
        "per_target_r2_all_steps": per_target_r2_all_steps,
        "overall_r2_final_step": overall_r2_final,
        "per_target_r2_final_step": per_target_r2_final
    }
    return metrics

def train_one_epoch_teacher_forcing(model, dataloader, loss_fn, optimizer, device, teacher_forcing_ratio=0.5):
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)          # shape: (batch, seq_len, input_dim)
        y_batch = y_batch.to(device)          # shape: (batch, seq_len, output_dim)

        batch_size, seq_len, _ = X_batch.shape
        input_seq = X_batch.clone()

        outputs = torch.zeros_like(y_batch)

        # initialize first input step
        x_t = input_seq[:, 0, :]

        hidden = None  # LSTM hidden state will be managed by PyTorch automatically

        for t in range(seq_len):
            x_t = x_t.unsqueeze(1) if x_t.dim() == 2 else x_t
            lstm_out, hidden = model.lstm(x_t, hidden)
            y_t = model.head(lstm_out.squeeze(1))       # (batch, output_dim)
            outputs[:, t, :] = y_t

            # decide whether to use teacher forcing
            if t + 1 < seq_len:
                if np.random.rand() < teacher_forcing_ratio:
                    # use ground truth for next input
                    x_t = input_seq[:, t + 1, :].clone()
                else:
                    # use model's own prediction
                    x_t = x_t.clone()
                    # if input_dim != output_dim, you may need to insert predicted outputs in the correct slice

        loss = loss_fn(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        preds = model(X_batch)
        loss = loss_fn(preds, y_batch)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate_model(model, dataloader, loss_fn, device, target_names):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            preds = model(X_batch)
            loss = loss_fn(preds, y_batch)

            total_loss += loss.item()
            all_preds.append(preds.cpu())
            all_targets.append(y_batch.cpu())

    avg_loss = total_loss / len(dataloader)

    preds_tensor = torch.cat(all_preds, dim=0)
    targets_tensor = torch.cat(all_targets, dim=0)

    y_pred = preds_tensor.numpy()
    y_true = targets_tensor.numpy()

    metrics = compute_sequence_metrics(y_true, y_pred, target_names)

    return avg_loss, metrics, preds_tensor, targets_tensor


def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs, target_names, device):
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_r2_all_steps": [],
        "val_r2_final_step": [],
        "val_r2_per_target_all_steps": [],
        "val_r2_per_target_final_step": []
    }

    for epoch in range(epochs):
        #train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        train_loss = train_one_epoch_teacher_forcing(model, train_loader, loss_fn, optimizer, device, teacher_forcing_ratio=0.9)
        val_loss, metrics, _, _ = evaluate_model(model, test_loader, loss_fn, device, target_names)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_r2_all_steps"].append(metrics["overall_r2_all_steps"])
        history["val_r2_final_step"].append(metrics["overall_r2_final_step"])
        history["val_r2_per_target_all_steps"].append(metrics["per_target_r2_all_steps"])
        history["val_r2_per_target_final_step"].append(metrics["per_target_r2_final_step"])

        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"  Train Loss:             {train_loss:.6f}")
        print(f"  Val Loss:               {val_loss:.6f}")
        print(f"  Val R² (all steps):     {metrics['overall_r2_all_steps']:.6f}")
        print(f"  Val R² (final step):    {metrics['overall_r2_final_step']:.6f}")

        print("  Per-target Val R² (all steps):")
        for name, r2_val in zip(target_names, metrics["per_target_r2_all_steps"]):
            print(f"    {name:>10s}: {r2_val:.6f}")

        print("  Per-target Val R² (final step):")
        for name, r2_val in zip(target_names, metrics["per_target_r2_final_step"]):
            print(f"    {name:>10s}: {r2_val:.6f}")

    return history


# =========================================================
# TRAIN MODEL
# =========================================================
history = train_model(
    model=model,
    train_loader=train_loader,
    test_loader=test_loader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    epochs=EPOCHS,
    target_names=target_names,
    device=DEVICE
)


# =========================================================
# FINAL EVALUATION
# =========================================================
final_val_loss, final_metrics, final_preds, final_targets = evaluate_model(
    model=model,
    dataloader=test_loader,
    loss_fn=loss_fn,
    device=DEVICE,
    target_names=target_names
)

print("\n" + "=" * 70)
print("FINAL VALIDATION METRICS")
print("=" * 70)
print(f"Final Validation Loss:            {final_val_loss:.6f}")
print(f"Final Overall R² (all steps):     {final_metrics['overall_r2_all_steps']:.6f}")
print(f"Final Overall R² (final step):    {final_metrics['overall_r2_final_step']:.6f}")

print("\nFinal Per-Target R² (all steps):")
for name, r2_val in zip(target_names, final_metrics["per_target_r2_all_steps"]):
    print(f"  {name:>10s}: {r2_val:.6f}")

print("\nFinal Per-Target R² (final step):")
for name, r2_val in zip(target_names, final_metrics["per_target_r2_final_step"]):
    print(f"  {name:>10s}: {r2_val:.6f}")


# =========================================================
# REAL-UNIT EVALUATION
# =========================================================
final_preds_np = final_preds.numpy()
final_targets_np = final_targets.numpy()

final_preds_real = inverse_transform_outputs_3d(final_preds_np, output_scalers, target_names)
final_targets_real = inverse_transform_outputs_3d(final_targets_np, output_scalers, target_names)

real_metrics = compute_sequence_metrics(final_targets_real, final_preds_real, target_names)

print("\n" + "=" * 70)
print("FINAL VALIDATION METRICS IN REAL CONCENTRATION UNITS")
print("=" * 70)
print(f"Overall R² in real units (all steps):  {real_metrics['overall_r2_all_steps']:.6f}")
print(f"Overall R² in real units (final step): {real_metrics['overall_r2_final_step']:.6f}")

print("\nPer-Target R² in real units (all steps):")
for name, r2_val in zip(target_names, real_metrics["per_target_r2_all_steps"]):
    print(f"  {name:>10s}: {r2_val:.6f}")

print("\nPer-Target R² in real units (final step):")
for name, r2_val in zip(target_names, real_metrics["per_target_r2_final_step"]):
    print(f"  {name:>10s}: {r2_val:.6f}")


# =========================================================
# TRAINING CURVES
# =========================================================
def plot_training_history(history):
    epochs_range = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, history["train_loss"], label="Train Loss")
    plt.plot(epochs_range, history["val_loss"], label="Validation Loss", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("MAE Loss")
    plt.title("Training / Validation Loss")
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, history["val_r2_all_steps"], label="Val R² (all steps)")
    plt.xlabel("Epoch")
    plt.ylabel("R²")
    plt.title("Validation R² Across Entire Predicted Sequence")
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(epochs_range, history["val_r2_final_step"], label="Val R² (final step)")
    plt.xlabel("Epoch")
    plt.ylabel("R²")
    plt.title("Validation R² at Final Predicted Timepoint")
    plt.legend()

    plt.tight_layout()
    plt.show()


plot_training_history(history)


# =========================================================
# OPTIONAL: VISUALIZE EXAMPLE TRAJECTORIES
# =========================================================
def plot_example_trajectories(y_true_real, y_pred_real, times, target_names, n_examples=3):
    """
    y_true_real, y_pred_real: shape (N_test, seq_len, output_dim)
    times should correspond to predicted target times (i.e. unique_times[1:])
    """
    n_examples = min(n_examples, y_true_real.shape[0])

    for ex in range(n_examples):
        for j, target in enumerate(target_names):
            plt.figure(figsize=(7, 4))
            plt.plot(times, y_true_real[ex, :, j], marker="o", label="True")
            plt.plot(times, y_pred_real[ex, :, j], marker="o", linestyle="--", label="Predicted")
            plt.title(f"Trajectory Example {ex + 1}: {target}")
            plt.xlabel("Time")
            plt.ylabel(f"{target} concentration")
            plt.legend()
            plt.tight_layout()
            plt.show()


predicted_times = unique_times[1:]
plot_example_trajectories(
    y_true_real=final_targets_real,
    y_pred_real=final_preds_real,
    times=predicted_times,
    target_names=target_names[:3],
    n_examples=2
)


# =========================================================
# COMBINED SHAP HELPER FUNCTION
# =========================================================

class UnifiedShapWrapper(nn.Module):
    """
    Wrapper for LSTM SHAP explanations.
    Modes:
        - 'all_timesteps': explain full input sequence → final output
        - 'initial_timesteps': explain only initial t0 → final output
    """
    def __init__(self, base_model, seq_len, input_dim, mode="all_timesteps"):
        super().__init__()
        self.base_model = base_model
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.mode = mode.lower()
        assert self.mode in ["all_timesteps", "initial_timesteps"], "mode must be 'all_timesteps' or 'initial_timesteps'"

    def forward(self, x):
        """
        x shape:
            - all_timesteps: (batch, seq_len*input_dim) flattened
            - initial_timesteps: (batch, input_dim)
        """
        if self.mode == "all_timesteps":
            x_seq = x.view(-1, self.seq_len, self.input_dim)
        else:  # initial_timesteps
            x_seq = x.unsqueeze(1).repeat(1, self.seq_len, 1)

        preds_seq = self.base_model(x_seq)
        return preds_seq[:, -1, :]  # final step only


def compute_shap_unified(
    model,
    X_train_seq,
    X_test_seq,
    feature_names,
    target_names,
    mode="all_timesteps",
    background_size=64,
    explain_size=64,
):
    """
    Unified SHAP computation for LSTM models.
    Parameters
    ----------
    mode : str
        'all_timesteps' = flattened full sequence → final output
        'initial_timesteps' = t0 initial concentrations → final output

    Returns
    -------
    shap_values : ndarray (n_samples, input_dim, n_targets)
    X_for_color : ndarray (n_samples, input_dim)
    """
    seq_len = X_train_seq.shape[1]
    input_dim = X_train_seq.shape[2]

    wrapper = UnifiedShapWrapper(model, seq_len=seq_len, input_dim=input_dim, mode=mode).to(DEVICE)
    wrapper.eval()

    background_n = min(background_size, len(X_train_seq))
    explain_n = min(explain_size, len(X_test_seq))

    if mode == "all_timesteps":
        background = torch.tensor(
            X_train_seq[:background_n].reshape(background_n, -1),
            dtype=torch.float32,
            device=DEVICE
        )
        explain = torch.tensor(
            X_test_seq[:explain_n].reshape(explain_n, -1),
            dtype=torch.float32,
            device=DEVICE
        )
    else:  # initial_timesteps
        background = torch.tensor(
            X_train_seq[:background_n, 0, :],
            dtype=torch.float32,
            device=DEVICE
        )
        explain = torch.tensor(
            X_test_seq[:explain_n, 0, :],
            dtype=torch.float32,
            device=DEVICE
        )

    prev_cudnn_state = torch.backends.cudnn.enabled
    try:
        torch.backends.cudnn.enabled = False
        explainer = shap.GradientExplainer(wrapper, background)
        shap_values = explainer.shap_values(explain)
    finally:
        torch.backends.cudnn.enabled = prev_cudnn_state

    if isinstance(shap_values, list):
        shap_values = np.stack(shap_values, axis=-1)

    shap_values = np.asarray(shap_values)

    if shap_values.ndim != 3:
        raise RuntimeError(f"Unexpected SHAP output shape: {shap_values.shape}")

    # For coloring in summary plots
    if mode == "all_timesteps":
        X_for_color = explain.detach().cpu().numpy().reshape(explain_n, seq_len, input_dim).mean(axis=1)
    else:
        X_for_color = explain.detach().cpu().numpy()

    return shap_values, X_for_color

# =========================================================
# SHAP SUMMARY TABLE (UNIFIED)
# =========================================================
def shap_summary_table(
    shap_values,
    X_for_color,
    targets,
    feature_names,
    cc_feature_names=CYTOKINE_CHEMOKINE_FEATURES,
    tf_feature_names=TF_FEATURES
):
    """
    Generate summary tables of SHAP values per target, separating cytokines/chemokines
    and transcription factors.

    shap_values: (n_samples, n_features, n_targets)
    X_for_color: (n_samples, n_features) for coloring/plotting
    targets: list of target names
    feature_names: list of feature names
    """
    feature_names = list(feature_names)

    cc_idx = [feature_names.index(f) for f in cc_feature_names if f in feature_names]
    tf_idx = [feature_names.index(f) for f in tf_feature_names if f in feature_names]

    cc_names = [feature_names[j] for j in cc_idx]
    tf_names = [feature_names[j] for j in tf_idx]

    cc_summary = {}
    tf_summary = {}

    for i, target in enumerate(targets):
        sv_target = shap_values[:, :, i]  # (n_samples, n_features)

        # Cytokines / chemokines
        sv_cc = sv_target[:, cc_idx]
        cc_summary[target] = pd.DataFrame({
            "Feature": cc_names,
            "Mean_Abs_SHAP": np.mean(np.abs(sv_cc), axis=0),
            "Min_SHAP": np.min(sv_cc, axis=0),
            "Max_SHAP": np.max(sv_cc, axis=0)
        }).sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

        # Transcription factors
        sv_tf = sv_target[:, tf_idx]
        tf_summary[target] = pd.DataFrame({
            "Feature": tf_names,
            "Mean_Abs_SHAP": np.mean(np.abs(sv_tf), axis=0),
            "Min_SHAP": np.min(sv_tf, axis=0),
            "Max_SHAP": np.max(sv_tf, axis=0)
        }).sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)

    return cc_summary, tf_summary


# =========================================================
# SHAP PLOTTING (DUAL, UNIFIED)
# =========================================================
def shap_explainer_dual(
    shap_values,
    X_for_color,
    targets,
    feature_names,
    mode="all_timesteps",
    cc_feature_names=CYTOKINE_CHEMOKINE_FEATURES,
    tf_feature_names=TF_FEATURES,
    save_plots=SAVE_SHAP_PLOTS,
    save_dir=SHAP_PLOT_DIR
):
    """
    Generate SHAP summary plots for LSTM models.

    mode: "all_timesteps" -> history → final
          "initial_timesteps" -> initial concentration → final

    shap_values: (n_samples, n_features, n_targets)
    X_for_color: (n_samples, n_features)
    """
    feature_names = list(feature_names)

    cc_idx = [feature_names.index(f) for f in cc_feature_names if f in feature_names]
    tf_idx = [feature_names.index(f) for f in tf_feature_names if f in feature_names]

    cc_names = [feature_names[j] for j in cc_idx]
    tf_names = [feature_names[j] for j in tf_idx]

    if save_plots:
        os.makedirs(save_dir, exist_ok=True)

    title_mode = "Full sequence → final" if mode == "all_timesteps" else "Initial → final"
    print(f"\nGenerating LSTM SHAP plots ({title_mode})...")
    print(f"SHAP values shape: {shap_values.shape}")

    for i, target in enumerate(targets):
        sv_target = shap_values[:, :, i]

        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        fig.suptitle(f"LSTM SHAP Feature Importance ({title_mode}) - {target}",
                     fontsize=16, fontweight="bold")

        plt.sca(axes[0])
        shap.summary_plot(
            sv_target[:, cc_idx],
            X_for_color[:, cc_idx],
            feature_names=cc_names,
            show=False,
            plot_size=None
        )
        axes[0].set_title("Cytokines & Chemokines")
        axes[0].set_xlabel("SHAP value magnitude")

        plt.sca(axes[1])
        shap.summary_plot(
            sv_target[:, tf_idx],
            X_for_color[:, tf_idx],
            feature_names=tf_names,
            show=False,
            plot_size=None
        )
        axes[1].set_title("Transcription Factors")
        axes[1].set_xlabel("SHAP value magnitude")

        plt.tight_layout()

        if save_plots:
            safe_name = target.replace("[", "").replace("]", "").replace("/", "_")
            save_path = os.path.join(save_dir, f"{safe_name}_lstm_shap_dual_{mode}.png")
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"Saved SHAP plot: {save_path}")

        plt.show()

# Explain full sequence → final prediction
shap_seq_agg, X_seq_color = compute_shap_unified(
    model, X_train, X_test, feature_names, target_names,
    mode="all_timesteps"
)

cc_summary, tf_summary = shap_summary_table(
    shap_values=shap_seq_agg,
    X_for_color=X_seq_color,
    targets=target_names,
    feature_names=feature_names
)

shap_explainer_dual(
    shap_values=shap_seq_agg,
    X_for_color=X_seq_color,
    targets=target_names,
    feature_names=feature_names,
    mode="all_timesteps",
    save_plots=SAVE_SHAP_PLOTS
)

# Explain initial t0 → final prediction
shap_init_agg, X_init_color = compute_shap_unified(
    model, X_train, X_test, feature_names, target_names,
    mode="initial_timesteps"
)

cc_summary, tf_summary = shap_summary_table(
    shap_values=shap_init_agg,
    X_for_color=X_init_color,
    targets=target_names,
    feature_names=feature_names
)

shap_explainer_dual(
    shap_values=shap_init_agg,
    X_for_color=X_init_color,
    targets=target_names,
    feature_names=feature_names,
    mode="initial_timesteps",
    save_plots=SAVE_SHAP_PLOTS
)


# =========================================================
# ICE + PDP FOR LSTM
# =========================================================
def plot_ice_pdp_real_units_lstm(
    model,
    X_seq_scaled,
    feature_names,
    features_to_plot,
    output_names,
    output_scalers,
    input_scaler,   # <-- pass this in explicitly
    target_idx,
    grid_points=40,
    ci_z=1.96,
    max_ice_curves=100,
    vary_mode="all_timesteps",
    save_plots=False,
    save_dir=ICE_PDP_DIR
):
    """
    ICE/PDP for sequence model using REAL INPUT CONCENTRATION on the x-axis
    and REAL OUTPUT CONCENTRATION on the y-axis.
    """
    model.eval()
    X_np = X_seq_scaled.copy()

    if max_ice_curves is not None and max_ice_curves < len(X_np):
        rng = np.random.default_rng(RANDOM_STATE)
        selected_idx = rng.choice(len(X_np), size=max_ice_curves, replace=False)
        X_np = X_np[selected_idx]

    target_name = output_names[target_idx]
    output_scaler = output_scalers[target_name]

    if save_plots:
        os.makedirs(save_dir, exist_ok=True)

    # input scaler parameters for inverse-transforming individual features
    input_means = input_scaler.mean_
    input_scales = input_scaler.scale_

    def scaled_to_real(feature_idx, scaled_values):
        return scaled_values * input_scales[feature_idx] + input_means[feature_idx]

    def real_to_scaled(feature_idx, real_values):
        return (real_values - input_means[feature_idx]) / input_scales[feature_idx]

    for feat in features_to_plot:
        if feat not in feature_names:
            continue

        feat_idx = feature_names.index(feat)

        # choose observed values in scaled space
        if vary_mode == "all_timesteps":
            observed_scaled = X_np[:, :, feat_idx].reshape(-1)
        elif vary_mode == "first_timestep":
            observed_scaled = X_np[:, 0, feat_idx]
        elif vary_mode == "last_timestep":
            observed_scaled = X_np[:, -1, feat_idx]
        else:
            raise ValueError(
                "vary_mode must be 'all_timesteps', 'first_timestep', or 'last_timestep'."
            )

        # convert those observed values back to real concentration units
        observed_real = scaled_to_real(feat_idx, observed_scaled)

        # define grid in real concentration units
        grid_real = np.linspace(
            np.min(observed_real),
            np.max(observed_real),
            grid_points
        )

        ice_curves = []

        for i in range(len(X_np)):
            x_instance = X_np[i].copy()
            preds_real = []

            for val_real in grid_real:
                x_temp = x_instance.copy()

                # convert real concentration to scaled value for model input
                val_scaled = real_to_scaled(feat_idx, val_real)

                if vary_mode == "all_timesteps":
                    x_temp[:, feat_idx] = val_scaled
                elif vary_mode == "first_timestep":
                    x_temp[0, feat_idx] = val_scaled
                elif vary_mode == "last_timestep":
                    x_temp[-1, feat_idx] = val_scaled

                x_torch = torch.tensor(
                    x_temp,
                    dtype=torch.float32,
                    device=DEVICE
                ).unsqueeze(0)

                with torch.no_grad():
                    pred_scaled_final = model(x_torch).cpu().numpy()[0, -1, target_idx]

                pred_real = output_scaler.inverse_transform([[pred_scaled_final]])[0, 0]
                preds_real.append(pred_real)

            ice_curves.append(preds_real)

        ice_curves = np.array(ice_curves)
        pdp = ice_curves.mean(axis=0)

        if ice_curves.shape[0] > 1:
            std = ice_curves.std(axis=0, ddof=1)
            sem = std / np.sqrt(ice_curves.shape[0])
        else:
            sem = np.zeros_like(pdp)

        lower = pdp - ci_z * sem
        upper = pdp + ci_z * sem

        plt.figure(figsize=(8, 5))

        for curve in ice_curves:
            plt.plot(grid_real, curve, color="black", alpha=0.15)

        plt.plot(grid_real, pdp, linewidth=2.2, label="PDP")
        plt.fill_between(grid_real, lower, upper, alpha=0.25, label="95% CI")

        if vary_mode == "all_timesteps":
            title_suffix = "all input timesteps"
        elif vary_mode == "first_timestep":
            title_suffix = "first input timestep"
        else:
            title_suffix = "last input timestep"

        plt.title(f"LSTM ICE + PDP: Predicted final {target_name} vs input {feat} ({title_suffix})")
        plt.xlabel(f"{feat} concentration")
        plt.ylabel(f"Predicted final {target_name} concentration")
        plt.legend()
        plt.tight_layout()

        if save_plots:
            safe_target = target_name.replace("[", "").replace("]", "").replace("/", "_")
            safe_feat = feat.replace("[", "").replace("]", "").replace("/", "_")
            save_path = os.path.join(save_dir, f"{safe_target}_vs_{safe_feat}_lstm_ice_pdp.png")
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"Saved ICE/PDP plot: {save_path}")

        plt.show()


# =========================================================
# RUN ICE + PDP FOR SHAP FEATURE GROUPS
# =========================================================
features_to_plot = []
for feat in CYTOKINE_CHEMOKINE_FEATURES + TF_FEATURES:
    if feat in feature_names and feat not in features_to_plot:
        features_to_plot.append(feat)

print("\nGenerating LSTM ICE + PDP plots in real concentration units...")
print("Features included:", features_to_plot)
for target_idx, target_name in enumerate(target_names):
    print(f"\nTarget: {target_name}")
    plot_ice_pdp_real_units_lstm(
        model=model,
        X_seq_scaled=X_test,
        feature_names=feature_names,
        features_to_plot=features_to_plot,
        output_names=target_names,
        output_scalers=output_scalers,
        input_scaler=input_scaler,   # <-- add this
        target_idx=target_idx,
        grid_points=40,
        ci_z=1.96,
        max_ice_curves=100,
        vary_mode="all_timesteps",
        save_plots=SAVE_ICE_PDP_PLOTS
    )


# =========================================================
# OPTIONAL: TABULAR PER-TARGET R2 SUMMARY
# =========================================================
r2_summary_df = pd.DataFrame({
    "Target": target_names,
    "R2_All_Steps_Scaled": final_metrics["per_target_r2_all_steps"],
    "R2_Final_Step_Scaled": final_metrics["per_target_r2_final_step"],
    "R2_All_Steps_RealUnits": real_metrics["per_target_r2_all_steps"],
    "R2_Final_Step_RealUnits": real_metrics["per_target_r2_final_step"]
}).sort_values("R2_Final_Step_RealUnits", ascending=False).reset_index(drop=True)

print("\nPer-target R² summary:")
print(r2_summary_df)
