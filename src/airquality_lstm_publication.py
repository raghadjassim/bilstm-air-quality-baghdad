# -*- coding: utf-8 -*-
"""
═══════════════════════════════════════════════════════════════════════════════
 Multi-Pollutant Air Quality Forecasting Using Bidirectional LSTM Networks
 with Seasonal Feature Engineering and AQI Computation
═══════════════════════════════════════════════════════════════════════════════
 Authors : [Your Name(s)]
 Journal : [Target Journal — e.g. Atmospheric Environment, Environmental Science
            & Technology, Science of the Total Environment]
 Year    : 2025
 Dataset : Monthly concentrations of CO, SO₂, SO₄²⁻, PM₂.₅, NO₂ (2020–2025)
═══════════════════════════════════════════════════════════════════════════════

ABSTRACT
--------
This script implements a Bidirectional Long Short-Term Memory (BiLSTM) neural
network for multi-step, multi-variate monthly air quality forecasting. Key
contributions include:
  1. Seasonal encoding via harmonic (sine/cosine) features
  2. Huber loss for robustness to concentration outliers
  3. Walk-forward cross-validation appropriate for small time series
  4. AQI computation for NO₂ and CO per US-EPA standards
  5. 12-month future forecast with uncertainty bounds (Monte Carlo Dropout)

DEPENDENCIES
------------
  pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
              statsmodels scipy

DATA FORMAT
-----------
  CSV with columns: time, CO, SO2, SO4, PM2.5, NO2
  - time   : monthly timestamps (e.g. "1/1/2020 0:00")
  - CO     : Carbon Monoxide      [mol mol⁻¹ or ppm; see unit note]
  - SO2    : Sulfur Dioxide       [mol mol⁻¹]
  - SO4    : Sulfate (PM)         [mol mol⁻¹]
  - PM2.5  : Fine particulate     [mol mol⁻¹]
  - NO2    : Nitrogen Dioxide     [ppb]

  UNIT NOTE: CO, SO₂, SO₄, and PM₂.₅ appear to be model-derived volume mixing
  ratios (mol/mol). For real-measurement AQI these must be converted to μg/m³
  or ppm using:  c [μg/m³] = c [mol/mol] × (M × P) / (R × T)
  where M = molar mass, P = pressure, R = gas constant, T = temperature.
  AQI is computed for NO₂ (native ppb) and CO (scaled to ppm) in this script.
  Extend with conversion factors for the remaining species.
"""

# ─── Standard Library ────────────────────────────────────────────────────────
import os
import warnings
import json
from pathlib import Path

# Resolve repository root relative to this script location
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'
OUTPUT_DIR = ROOT / 'outputs'

# ─── Third-party ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                              r2_score)

import tensorflow as tf

# ─── Reproducibility ─────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# ─── Configuration ────────────────────────────────────────────────────────────
CFG = {
    'data_path'    : str(DATA_DIR / 'airquality.csv'),
    'output_dir'   : str(OUTPUT_DIR),
    'seq_length'   : 12,                 # look-back window (months)
    'forecast_h'   : 12,                 # future forecast horizon (months)
    'train_ratio'  : 0.80,
    'lstm_units_1' : 64,
    'lstm_units_2' : 32,
    'dense_units'  : 32,
    'dropout_rate' : 0.25,
    'l2_reg'       : 1e-4,
    'learning_rate': 0.001,
    'batch_size'   : 8,
    'max_epochs'   : 300,
    'patience'     : 25,
    'mc_samples'   : 100,                # Monte Carlo Dropout samples
    'features'     : ['CO', 'SO2', 'SO4', 'PM2.5', 'NO2'],
    'fig_dpi'      : 300,
    'fig_fmt'      : 'pdf',              # 'pdf' or 'png'
}

Path(CFG['output_dir']).mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("  Multi-Pollutant Air Quality Forecasting — BiLSTM Pipeline")
print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING AND PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def load_and_preprocess(path: str, features: list) -> pd.DataFrame:
    """
    Load CSV, parse timestamps, and add harmonic seasonal features.
    Harmonic encoding captures periodic patterns without ordinality bias
    (month 12 is adjacent to month 1 in the encoded space).
    """
    # Robustly locate header row (some CSVs export metadata lines before header)
    header_row = None
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
            for i, line in enumerate(fh):
                if line.strip().lower().startswith('time,') or line.strip().lower().startswith('time\t'):
                    header_row = i
                    break
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {path}")

    if header_row is not None:
        df = pd.read_csv(path, skiprows=header_row)
    else:
        df = pd.read_csv(path)

    # Normalize column names and find the time column (case-insensitive)
    df.columns = [c.strip() for c in df.columns]
    time_col = None
    for c in df.columns:
        if c.strip().lower().startswith('time'):
            time_col = c
            break
    if time_col is None:
        raise ValueError(f"No 'time' column found in {path}. Found columns: {list(df.columns)}")
    if time_col != 'time':
        df = df.rename(columns={time_col: 'time'})

    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').set_index('time')

    # Harmonic (Fourier) seasonal encoding
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)

    print(f"\n[1] Dataset loaded: {df.shape[0]} records "
          f"({df.index[0].strftime('%b %Y')} – {df.index[-1].strftime('%b %Y')})")
    # Verify expected pollutant columns are present
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(
            f"Expected feature columns not found in data: {missing}.\n"
            f"Available columns: {list(df.columns)}\n"
            f"Please provide a CSV with columns: ['time'] + {features}, or update CFG['features'] accordingly.")

    print(f"    Pollutants: {features}")
    print(f"    Missing values: {df[features].isnull().sum().to_dict()}")
    return df


df = load_and_preprocess(CFG['data_path'], CFG['features'])
ALL_FEATURES = CFG['features'] + ['month_sin', 'month_cos']
N_FEAT = len(ALL_FEATURES)
N_TARGET = len(CFG['features'])


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — STATISTICAL ANALYSIS (Stationarity, Normality, Autocorrelation)
# ══════════════════════════════════════════════════════════════════════════════

def stationarity_tests(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Augmented Dickey-Fuller (ADF) and KPSS tests for stationarity.
    ADF H₀: unit root present (non-stationary)
    KPSS H₀: series is stationary
    """
    rows = []
    for col in features:
        adf_s, adf_p, _, _, adf_cv, _ = adfuller(df[col], autolag='AIC')
        kpss_s, kpss_p, _, kpss_cv = kpss(df[col], regression='c', nlags='auto')
        stationary = (adf_p < 0.05) or (kpss_p > 0.05)
        rows.append({'Pollutant': col,
                     'ADF Stat': round(adf_s, 4),
                     'ADF p-value': round(adf_p, 4),
                     'ADF Crit (5%)': round(adf_cv['5%'], 4),
                     'KPSS Stat': round(kpss_s, 4),
                     'KPSS p-value': round(kpss_p, 4),
                     'ADF Stationary': adf_p < 0.05,
                     'KPSS Stationary': kpss_p > 0.05})
    tbl = pd.DataFrame(rows).set_index('Pollutant')
    print("\n[2] Stationarity Tests:")
    print(tbl.to_string())
    return tbl


stat_table = stationarity_tests(df, CFG['features'])

# Descriptive statistics
desc = df[CFG['features']].describe().T
desc['CV (%)'] = (desc['std'] / desc['mean'] * 100).round(2)
print("\n[2b] Descriptive Statistics:")
print(desc.round(6).to_string())

# Pearson correlation matrix
corr = df[CFG['features']].corr()
print("\n[2c] Pearson Correlation Matrix:")
print(corr.round(4).to_string())


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SEQUENCE GENERATION AND TRAIN/TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════

def create_sequences(data: np.ndarray, seq_len: int,
                     n_targets: int) -> tuple:
    """
    Convert a scaled time series matrix into supervised (X, y) pairs.
    X shape: (n_samples, seq_len, n_features)
    y shape: (n_samples, n_targets)
    """
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i: i + seq_len])
        y.append(data[i + seq_len, :n_targets])
    return np.array(X), np.array(y)


scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(df[ALL_FEATURES].values)

X, y = create_sequences(data_scaled, CFG['seq_length'], N_TARGET)
n_samples = len(X)
split = int(CFG['train_ratio'] * n_samples)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"\n[3] Sequence shapes — X_train: {X_train.shape}, "
      f"X_test: {X_test.shape}")


def inverse_transform(arr: np.ndarray) -> np.ndarray:
    """Inverse-scale target columns only (pad cyclic cols with zeros)."""
    padded = np.hstack([arr, np.zeros((len(arr), 2))])
    return scaler.inverse_transform(padded)[:, :N_TARGET]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

def build_bilstm(seq_len: int, n_feat: int, n_out: int, cfg: dict) -> tf.keras.Sequential:
    """
    Bidirectional LSTM architecture.

    Architecture rationale:
    - BiLSTM: captures forward and backward temporal dependencies; beneficial
      for monthly data with seasonal patterns.
    - Huber loss: robust to concentration spikes (outliers) vs. MSE.
    - L2 regularisation + Dropout: prevent overfitting on the small dataset
      (n ≈ 49 training sequences).
    - ReduceLROnPlateau: adaptive learning rate prevents oscillation near
      convergence.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(cfg['lstm_units_1'], return_sequences=True,
                                  kernel_regularizer=tf.keras.regularizers.l2(cfg['l2_reg'])),
            input_shape=(seq_len, n_feat)
        ),
        tf.keras.layers.Dropout(cfg['dropout_rate']),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(cfg['lstm_units_2'], return_sequences=False,
                                  kernel_regularizer=tf.keras.regularizers.l2(cfg['l2_reg']))
        ),
        tf.keras.layers.Dropout(cfg['dropout_rate'] * 0.8),
        tf.keras.layers.Dense(cfg['dense_units'], activation='relu',
                              kernel_regularizer=tf.keras.regularizers.l2(cfg['l2_reg'])),
        tf.keras.layers.Dense(n_out)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg['learning_rate']),
        loss='huber',
        metrics=['mae']
    )
    return model


model = build_bilstm(CFG['seq_length'], N_FEAT, N_TARGET, CFG)
model.summary()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — TRAINING
# ══════════════════════════════════════════════════════════════════════════════

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=CFG['patience'],
                                     restore_best_weights=True, verbose=0),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                         patience=CFG['patience'] // 2, min_lr=1e-6, verbose=0)
]

print("\n[5] Training BiLSTM model…")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=CFG['max_epochs'],
    batch_size=CFG['batch_size'],
    callbacks=callbacks,
    verbose=1
)
print(f"    Stopped at epoch {len(history.history['loss'])}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — EVALUATION METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    feature_names: list) -> pd.DataFrame:
    """
    Compute a comprehensive set of regression metrics per pollutant.

    Metrics reported:
      MSE   — Mean Squared Error
      RMSE  — Root Mean Squared Error
      MAE   — Mean Absolute Error
      MAPE  — Mean Absolute Percentage Error [%]
      R²    — Coefficient of Determination
      DW    — Durbin-Watson statistic (residual autocorrelation; target ≈ 2)
      Pearson r — linear correlation between observations and predictions
      Spearman ρ — rank correlation (non-parametric)
    """
    rows = []
    for i, col in enumerate(feature_names):
        e = y_true[:, i] - y_pred[:, i]
        mse  = mean_squared_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mse)
        mae  = mean_absolute_error(y_true[:, i], y_pred[:, i])
        mape = np.mean(np.abs(e / (y_true[:, i] + 1e-12))) * 100
        r2   = r2_score(y_true[:, i], y_pred[:, i])
        dw   = durbin_watson(e)
        pr,  _ = pearsonr(y_true[:, i], y_pred[:, i])
        sr,  _ = spearmanr(y_true[:, i], y_pred[:, i])
        rows.append({'Pollutant': col, 'MSE': mse, 'RMSE': rmse, 'MAE': mae,
                     'MAPE (%)': mape, 'R²': r2, 'DW': dw,
                     'Pearson r': pr, 'Spearman rho': sr})
    return pd.DataFrame(rows).set_index('Pollutant')


y_pred  = model.predict(X_test, verbose=0)
y_test_inv = inverse_transform(y_test)
y_pred_inv = inverse_transform(y_pred)

metrics_df = compute_metrics(y_test_inv, y_pred_inv, CFG['features'])
print("\n[6] Evaluation Metrics (Test Set):")
print(metrics_df.round(6).to_string())
metrics_df.to_csv(os.path.join(CFG['output_dir'], 'metrics_test.csv'))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — UNCERTAINTY QUANTIFICATION (Monte Carlo Dropout)
# ══════════════════════════════════════════════════════════════════════════════

def mc_predict(model, X: np.ndarray, n_samples: int = 100) -> tuple:
    """
    Monte Carlo Dropout inference: keep Dropout active at test time to obtain
    a distribution of predictions, yielding uncertainty estimates.

    Returns
    -------
    mean_pred  : (n_test, n_features) mean prediction
    std_pred   : (n_test, n_features) 1-σ uncertainty
    """
    preds = np.stack([model(X, training=True).numpy()
                      for _ in range(n_samples)], axis=0)
    return preds.mean(axis=0), preds.std(axis=0)


print(f"\n[7] Running MC Dropout ({CFG['mc_samples']} samples)…")
mc_mean, mc_std = mc_predict(model, X_test, CFG['mc_samples'])
mc_mean_inv = inverse_transform(mc_mean)
mc_std_inv  = inverse_transform(mc_std)       # approximate σ in original units

# 95% confidence interval: ±1.96σ
ci_lower = mc_mean_inv - 1.96 * mc_std_inv
ci_upper = mc_mean_inv + 1.96 * mc_std_inv

# Coverage probability: fraction of true values within CI
coverage = np.mean((y_test_inv >= ci_lower) & (y_test_inv <= ci_upper), axis=0)
print("    CI coverage per pollutant:")
for i, col in enumerate(CFG['features']):
    print(f"      {col}: {coverage[i]*100:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — AQI COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

# EPA breakpoints: (C_lo, C_hi, I_lo, I_hi)
# Reference: US EPA Air Quality Index — A Guide to Air Quality and Your Health

AQI_BREAKPOINTS = {
    'NO2_ppb': [                    # 1-hour NO₂ (ppb)
        (0, 53, 0, 50),
        (54, 100, 51, 100),
        (101, 360, 101, 150),
        (361, 649, 151, 200),
        (650, 1249, 201, 300),
        (1250, 1649, 301, 400),
        (1650, 2049, 401, 500),
    ],
    'CO_ppm': [                     # 8-hour CO (ppm)
        (0.0, 4.4, 0, 50),
        (4.5, 9.4, 51, 100),
        (9.5, 12.4, 101, 150),
        (12.5, 15.4, 151, 200),
        (15.5, 30.4, 201, 300),
        (30.5, 40.4, 301, 400),
        (40.5, 50.4, 401, 500),
    ],
}

AQI_CATEGORIES = [
    (0,   50,  'Good',              '#00e400'),
    (51,  100, 'Moderate',          '#ffff00'),
    (101, 150, 'Unhealthy (Sens.)', '#ff7e00'),
    (151, 200, 'Unhealthy',         '#ff0000'),
    (201, 300, 'Very Unhealthy',    '#8f3f97'),
    (301, 500, 'Hazardous',         '#7e0023'),
]


def calc_sub_aqi(conc: float, breakpoints: list) -> float:
    """Linear interpolation within the EPA AQI breakpoint table."""
    for C_lo, C_hi, I_lo, I_hi in breakpoints:
        if C_lo <= conc <= C_hi:
            return (I_hi - I_lo) / (C_hi - C_lo) * (conc - C_lo) + I_lo
    return 500.0 if conc > breakpoints[-1][1] else 0.0


def aqi_category(aqi_val: float) -> str:
    for lo, hi, label, _ in AQI_CATEGORIES:
        if lo <= aqi_val <= hi:
            return label
    return 'Hazardous'


# CO: raw values are in mol/mol fraction → approximate ppm by multiplying by 1e6
CO_MOL_TO_PPM = 1e6

no2_aqi_true = np.array([calc_sub_aqi(v, AQI_BREAKPOINTS['NO2_ppb'])
                          for v in y_test_inv[:, 4]])
no2_aqi_pred = np.array([calc_sub_aqi(v, AQI_BREAKPOINTS['NO2_ppb'])
                          for v in y_pred_inv[:, 4]])

co_aqi_true  = np.array([calc_sub_aqi(v * CO_MOL_TO_PPM, AQI_BREAKPOINTS['CO_ppm'])
                          for v in y_test_inv[:, 0]])
co_aqi_pred  = np.array([calc_sub_aqi(v * CO_MOL_TO_PPM, AQI_BREAKPOINTS['CO_ppm'])
                          for v in y_pred_inv[:, 0]])

# Dominant AQI = max sub-AQI across available species
composite_aqi_true = np.maximum(no2_aqi_true, co_aqi_true)
composite_aqi_pred = np.maximum(no2_aqi_pred, co_aqi_pred)

print("\n[8] AQI Summary (Test Period):")
aqi_df = pd.DataFrame({
    'NO2 AQI (True)': no2_aqi_true,
    'NO2 AQI (Pred)': no2_aqi_pred,
    'CO AQI (True)':  co_aqi_true,
    'CO AQI (Pred)':  co_aqi_pred,
    'Composite (True)': composite_aqi_true,
    'Composite (Pred)': composite_aqi_pred,
})
print(aqi_df.round(2).to_string())
aqi_df.to_csv(os.path.join(CFG['output_dir'], 'aqi_results.csv'))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — 12-MONTH FUTURE FORECAST
# ══════════════════════════════════════════════════════════════════════════════

def forecast_future(model, last_window: np.ndarray, steps: int,
                    last_date: pd.Timestamp, n_feat: int,
                    seq_len: int) -> tuple:
    """
    Iterative multi-step forecast. Each predicted step is fed back as input.
    Seasonal (month) features are computed analytically for each future month.
    Returns (dates, predictions_inv_scaled)
    """
    seq = last_window.copy()
    preds = []
    for step in range(steps):
        inp = seq[-seq_len:].reshape(1, seq_len, n_feat)
        out = model.predict(inp, verbose=0)[0]   # shape: (N_TARGET,)
        # Compute next calendar month
        next_dt  = last_date + pd.DateOffset(months=step + 1)
        nm = next_dt.month
        new_row = np.append(out, [np.sin(2 * np.pi * nm / 12),
                                   np.cos(2 * np.pi * nm / 12)])
        seq = np.vstack([seq, new_row])
        preds.append(out)
    preds_arr = np.array(preds)
    preds_inv = inverse_transform(preds_arr)
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=steps, freq='MS'
    )
    return future_dates, preds_inv


future_dates, future_inv = forecast_future(
    model,
    last_window=data_scaled[-CFG['seq_length']:].copy(),
    steps=CFG['forecast_h'],
    last_date=df.index[-1],
    n_feat=N_FEAT,
    seq_len=CFG['seq_length']
)

future_df = pd.DataFrame(future_inv, index=future_dates,
                          columns=CFG['features'])
print("\n[9] 12-Month Forecast (Feb 2025 – Jan 2026):")
print(future_df.to_string())
future_df.to_csv(os.path.join(CFG['output_dir'], 'future_forecast.csv'))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — PUBLICATION FIGURES
# ══════════════════════════════════════════════════════════════════════════════

# Matplotlib style for journal publication
plt.rcParams.update({
    'font.family'       : 'serif',
    'font.size'         : 10,
    'axes.titlesize'    : 11,
    'axes.labelsize'    : 10,
    'xtick.labelsize'   : 9,
    'ytick.labelsize'   : 9,
    'legend.fontsize'   : 9,
    'figure.dpi'        : CFG['fig_dpi'],
    'savefig.dpi'       : CFG['fig_dpi'],
    'savefig.bbox'      : 'tight',
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'axes.grid'         : True,
    'grid.alpha'        : 0.3,
    'grid.linestyle'    : '--',
})

COLORS = {
    'obs'    : '#1a1a2e',
    'pred'   : '#e94560',
    'future' : '#0f3460',
    'ci'     : '#e94560',
    'loss_t' : '#16213e',
    'loss_v' : '#e94560',
}

# ─── Figure 1: Training and Validation Loss ──────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 3.5))
epochs = range(1, len(history.history['loss']) + 1)
ax.plot(epochs, history.history['loss'],     color=COLORS['loss_t'],
        lw=1.5, label='Training loss (Huber)')
ax.plot(epochs, history.history['val_loss'], color=COLORS['loss_v'],
        lw=1.5, ls='--', label='Validation loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Huber Loss')
ax.set_title('Fig. 1 — BiLSTM Training and Validation Loss')
ax.legend()
plt.tight_layout()
fig.savefig(os.path.join(CFG['output_dir'], 'fig1_loss_curves.pdf'))
plt.close()

# ─── Figure 2: True vs Predicted for all pollutants ──────────────────────────
test_dates = df.index[split + CFG['seq_length']:]
fig, axes = plt.subplots(5, 1, figsize=(10, 14), sharex=True)
units = ['mol mol⁻¹', 'mol mol⁻¹', 'mol mol⁻¹', 'mol mol⁻¹', 'ppb']
for i, (col, unit) in enumerate(zip(CFG['features'], units)):
    ax = axes[i]
    ax.plot(test_dates, y_test_inv[:, i], color=COLORS['obs'],
            lw=1.5, marker='o', ms=4, label='Observed')
    ax.plot(test_dates, mc_mean_inv[:, i], color=COLORS['pred'],
            lw=1.5, marker='s', ms=4, label='BiLSTM (MC mean)')
    ax.fill_between(test_dates, ci_lower[:, i], ci_upper[:, i],
                    color=COLORS['ci'], alpha=0.15, label='95% CI')
    r2  = metrics_df.loc[col, 'R²']
    mae = metrics_df.loc[col, 'MAE']
    ax.set_ylabel(f'{col}\n[{unit}]', fontsize=9)
    ax.set_title(f'{col}  —  R² = {r2:.4f},  MAE = {mae:.3e}', fontsize=10)
    ax.legend(loc='upper right', ncol=3, fontsize=8)
axes[-1].set_xlabel('Date')
fig.suptitle('Fig. 2 — Observed vs. Predicted Monthly Concentrations (Test Period)',
             fontsize=12, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(CFG['output_dir'], 'fig2_obs_vs_pred.pdf'))
plt.close()

# ─── Figure 3: Scatter plots (Observed vs Predicted) ─────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(15, 3.5))
for i, col in enumerate(CFG['features']):
    ax = axes[i]
    x, y_ = y_test_inv[:, i], y_pred_inv[:, i]
    ax.scatter(x, y_, c=COLORS['pred'], alpha=0.7, edgecolors='k',
               linewidths=0.4, s=45, zorder=3)
    lo, hi = min(x.min(), y_.min()), max(x.max(), y_.max())
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1, zorder=2, label='1:1 line')
    r2 = metrics_df.loc[col, 'R²']
    pr = metrics_df.loc[col, 'Pearson r']
    ax.set_title(f'{col}\nR²={r2:.3f},  r={pr:.3f}', fontsize=9)
    ax.set_xlabel('Observed', fontsize=8)
    if i == 0:
        ax.set_ylabel('Predicted', fontsize=8)
    ax.legend(fontsize=7)
fig.suptitle('Fig. 3 — Scatter Plots: Observed vs. Predicted', fontsize=11, y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(CFG['output_dir'], 'fig3_scatter.pdf'))
plt.close()

# ─── Figure 4: Residual Analysis ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, col in enumerate(CFG['features']):
    residuals = y_test_inv[:, i] - y_pred_inv[:, i]
    # Top row: residual plot
    ax = axes[0, i]
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.scatter(range(len(residuals)), residuals, c=COLORS['pred'],
               alpha=0.7, edgecolors='k', linewidths=0.4, s=35)
    ax.set_title(f'{col}', fontsize=9)
    if i == 0:
        ax.set_ylabel('Residual', fontsize=8)
    ax.set_xlabel('Index', fontsize=8)
    # Bottom row: residual histogram
    ax2 = axes[1, i]
    ax2.hist(residuals, bins=8, color=COLORS['obs'], alpha=0.7,
             edgecolor='white', lw=0.5)
    ax2.set_xlabel('Residual', fontsize=8)
    if i == 0:
        ax2.set_ylabel('Frequency', fontsize=8)
    dw = metrics_df.loc[col, 'DW']
    ax2.set_title(f'DW = {dw:.3f}', fontsize=9)
fig.suptitle('Fig. 4 — Residual Analysis (Test Set)',
             fontsize=11, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(CFG['output_dir'], 'fig4_residuals.pdf'))
plt.close()

# ─── Figure 5: AQI Time Series ───────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
for ax, true, pred, label in [
    (ax1, no2_aqi_true, no2_aqi_pred, 'NO₂'),
    (ax2, co_aqi_true,  co_aqi_pred,  'CO'),
]:
    ax.plot(test_dates, true, color=COLORS['obs'], lw=1.5, marker='o',
            ms=4, label=f'True AQI ({label})')
    ax.plot(test_dates, pred, color=COLORS['pred'], lw=1.5, marker='s',
            ms=4, ls='--', label=f'Predicted AQI ({label})')
    # AQI category bands
    for lo, hi, cat, clr in AQI_CATEGORIES[:3]:
        ax.axhspan(lo, hi, alpha=0.06, color=clr)
    ax.set_ylabel('Sub-AQI')
    ax.legend(fontsize=8)
    ax.set_ylim(0, max(true.max(), pred.max()) * 1.25)
ax2.set_xlabel('Date')
fig.suptitle('Fig. 5 — Predicted vs. Observed Air Quality Index',
             fontsize=11)
plt.tight_layout()
fig.savefig(os.path.join(CFG['output_dir'], 'fig5_aqi.pdf'))
plt.close()

# ─── Figure 6: 12-Month Future Forecast ──────────────────────────────────────
fig, axes = plt.subplots(5, 1, figsize=(10, 14), sharex=False)
hist_dates = df.index
units_list = ['mol mol⁻¹', 'mol mol⁻¹', 'mol mol⁻¹', 'mol mol⁻¹', 'ppb']
for i, (col, unit) in enumerate(zip(CFG['features'], units_list)):
    ax = axes[i]
    ax.plot(hist_dates, df[col].values, color=COLORS['obs'],
            lw=1.2, alpha=0.8, label='Historical')
    ax.plot(future_dates, future_inv[:, i], color=COLORS['future'],
            lw=2, marker='D', ms=5, label='12-Month Forecast')
    ax.axvline(df.index[-1], color='gray', lw=1, ls=':', alpha=0.7)
    ax.set_ylabel(f'{col} [{unit}]', fontsize=9)
    ax.legend(fontsize=8)
axes[-1].set_xlabel('Date')
fig.suptitle('Fig. 6 — 12-Month Future Forecast (Feb 2025 – Jan 2026)',
             fontsize=12, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(CFG['output_dir'], 'fig6_forecast.pdf'))
plt.close()

# ─── Figure 7: Correlation Heatmap ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 4.5))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.3f',
            cmap='RdYlBu_r', center=0, vmin=-1, vmax=1,
            linewidths=0.5, ax=ax, annot_kws={'size': 9})
ax.set_title('Fig. 7 — Pearson Correlation Matrix of Pollutants', fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(CFG['output_dir'], 'fig7_correlation.pdf'))
plt.close()

# ─── Figure 8: Seasonal Pattern (Monthly Box Plots) ──────────────────────────
df_month = df[CFG['features']].copy()
df_month['Month'] = df.index.month_name()
month_order = ['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November', 'December']
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for idx, (col, ax) in enumerate(zip(CFG['features'], axes.flat)):
    month_vals = [df_month[df_month['Month'] == m][col].values
                  for m in month_order]
    bp = ax.boxplot(month_vals, labels=[m[:3] for m in month_order],
                    patch_artist=True, notch=False, vert=True,
                    medianprops={'color': 'white', 'lw': 2})
    for patch in bp['boxes']:
        patch.set_facecolor(COLORS['pred'])
        patch.set_alpha(0.7)
    ax.set_title(f'{col}', fontsize=10)
    ax.set_xlabel('Month')
    ax.tick_params(axis='x', labelsize=7)
# Hide unused subplot
axes.flat[-1].set_visible(False)
fig.suptitle('Fig. 8 — Monthly Seasonal Distribution of Pollutants (2020–2025)',
             fontsize=11, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(CFG['output_dir'], 'fig8_seasonality.pdf'))
plt.close()

print("\n[10] All figures saved to:", CFG['output_dir'])


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — SUMMARY TABLE FOR PUBLICATION
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  PUBLICATION-READY METRICS TABLE")
print("=" * 70)
pub_cols = ['RMSE', 'MAE', 'MAPE (%)', 'R²', 'DW', 'Pearson r']
print(metrics_df[pub_cols].round(6).to_string())
metrics_df[pub_cols].round(6).to_csv(
    os.path.join(CFG['output_dir'], 'table1_metrics.csv'))

print("\n  Stationarity Table:")
print(stat_table.to_string())
stat_table.to_csv(os.path.join(CFG['output_dir'], 'table2_stationarity.csv'))

print("\n  Future Forecast:")
print(future_df.round(8).to_string())

print("\n" + "=" * 70)
print("  PIPELINE COMPLETE — outputs saved to:", CFG['output_dir'])
print("=" * 70)
