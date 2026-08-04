# 🌤️ Air Quality Forecasting & Interactive AQI Dashboard

An end-to-end Machine Learning pipeline utilizing **Bidirectional LSTM (BiLSTM)** networks to forecast multi-pollutant air quality concentrations and calculate US-EPA Air Quality Index (AQI) metrics for Baghdad (5-Year NASA Dataset).

---

## 📌 Project Overview

This project builds a time-series forecasting pipeline to predict atmospheric pollutants ($\text{NO}_2$, $\text{PM}_{2.5}$, $\text{CO}$, $\text{SO}_2$, $\text{SO}_4$) and map them directly to health hazard categories. It features automated model evaluation, statistical stationarity testing, and an interactive, single-file HTML5 dashboard.

### Key Highlights
- **Architecture:** Bidirectional LSTM with Huber Loss for outlier robustness.
- **Feature Engineering:** Circular harmonic sine/cosine month encoding for seasonality.
- **Uncertainty Quantification:** 95% confidence bounds via Monte Carlo (MC) Dropout.
- **Interactive UI:** Standalone web dashboard built with HTML5 & Chart.js.

---

## 📂 Repository Structure

```text
bilstm-air-quality-baghdad/
│
├── data/
│   └── airquality.csv                 # 5-Year NASA air quality dataset
│
├── outputs/
│   ├── airquality_dashboard.html      # Interactive visual dashboard
│   ├── metrics_test.csv               # Model performance metrics
│   ├── future_forecast.csv            # 12-month predictions
│   └── aqi_results.csv                # Derived US-EPA AQI values
│
├── src/
│   ├── airquality_lstm_publication.py # Model training & evaluation script
│   └── generate_dashboard.py          # Dashboard generation script
│
├── .gitignore                         # Git exclusion rules
├── requirements.txt                   # Project dependencies
└── README.md                          # Documentation
```
## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Train LSTM model and generate predictions
python src/airquality_lstm_publication.py

# Generate interactive dashboard
python src/generate_dashboard.py

Open the generated dashboard:

outputs/airquality_dashboard.html