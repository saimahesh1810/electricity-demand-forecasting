# Forecasting Weekly German Electricity Demand

This project forecasts weekly German electricity demand using statistical, machine-learning and neural-network models. It compares simple benchmarks with SARIMA, SARIMAX, Gradient Boosting and LSTM approaches over a fixed 104-week test period.

The main objective was to determine whether complex models provide a meaningful improvement over a strong seasonal-naïve benchmark.

## Project Summary

- **Target:** Weekly average German electricity demand in GW
- **Data source:** Open Power System Data Time Series package
- **Source file:** `time_series_60min_singleindex.csv`
- **Original frequency:** Hourly
- **Forecasting frequency:** Weekly
- **Final dataset:** 300 complete weeks
- **Training period:** 196 weeks
- **Test period:** Final 104 weeks
- **Seasonal period:** 52 weeks

The electricity-demand data were combined with:

- weekly temperature variables;
- heating and cooling degree days;
- German public-holiday variables;
- lagged demand features;
- rolling statistics;
- cyclical calendar features.

## Models Evaluated

The following forecasting approaches were compared:

1. Mean benchmark
2. Naïve benchmark
3. Seasonal-naïve benchmark
4. Drift benchmark
5. SARIMA
6. SARIMAX with temperature and holiday covariates
7. Gradient Boosting regression
8. LSTM neural network

All models were evaluated on the same final 104-week test period using:

- MAE
- RMSE
- MASE
- Bias
- sMAPE

## Main Results

| Rank | Model | MAE | RMSE | MASE | Bias | sMAPE |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Gradient Boosting | 2.3668 | 3.1334 | 1.6122 | 1.9051 | 4.3880 |
| 2 | Seasonal naïve | 2.4175 | 3.1549 | 1.6468 | 1.7720 | 4.4406 |
| 3 | LSTM | 2.7071 | 3.5358 | 1.8440 | 1.1743 | 4.9682 |
| 4 | SARIMA | 2.9901 | 3.6154 | 2.0368 | 1.1167 | 5.4633 |
| 5 | Naïve | 3.7319 | 4.3819 | 2.5421 | -0.2931 | 6.7709 |
| 6 | Mean | 3.7761 | 4.4046 | 2.5722 | 0.5346 | 6.8505 |
| 7 | Drift | 4.0330 | 4.7181 | 2.7472 | 0.6361 | 7.3093 |
| 8 | SARIMAX | 4.2480 | 5.0295 | 2.8936 | 4.2371 | 7.5852 |

Gradient Boosting achieved the lowest error, but its MAE improvement over seasonal naïve was only about 2.1%.

The 52-week lag was the dominant Gradient Boosting feature, confirming that annual persistence was the strongest predictive signal.

For operational use, seasonal naïve was recommended as the primary model because it was:

- almost as accurate as Gradient Boosting;
- transparent;
- inexpensive to maintain;
- independent of unknown future weather;
- easy to retrain and monitor.

Gradient Boosting was retained as a challenger model.

## Important Forecasting Design Decisions

### Chronological split

The final 104 weeks were reserved as an unseen test period. Random splitting was not used because it could allow future information to influence model training.

### Leakage prevention

Load lags and rolling statistics were constructed using only previous observations.

Rolling features were calculated from a one-week-shifted target:

```python
past_target = target_series.shift(1)
For recursive Gradient Boosting and LSTM forecasts, earlier predictions were inserted into later model inputs. Actual test-period demand was never used to create future lag features.

Conditional weather forecasts

Realised test-period weather was supplied to SARIMAX, Gradient Boosting and LSTM.

These results are therefore conditional forecasts. In an operational system, future observed temperature would need to be replaced by weather forecasts or weather scenarios.

Repository Structure

electricity-demand-forecasting/
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── outputs/
│   ├── figures/
│   ├── forecasts/
│   ├── metrics/
│   └── model_objects/
│
├── reports/
│   └── Electricity Demand Forecasting Report (24069723).pdf
│
├── scripts/
│   ├── download_data.py
│   ├── make_features.py
│   ├── evaluate_models.py
│   ├── run_sarima.py
│   ├── run_sarimax.py
│   ├── run_feature_model.py
│   ├── run_neural.py
│   └── run_pipeline.py
│
├── src/
│   └── electricity_demand/
│       ├── config.py
│       ├── data.py
│       ├── evaluation.py
│       ├── features.py
│       ├── pipeline.py
│       ├── plotting.py
│       └── models/
│           ├── benchmarks.py
│           ├── sarimax.py
│           ├── feature_models.py
│           └── neural.py
│
├── tests/
├── requirements.txt
├── environment.yml
├── pyproject.toml
└── README.md

Installation

Clone the repository:
git clone https://github.com/saimahesh1810electricity-demand-forecasting.git
cd electricity-demand-forecasting

Create a virtual environment:

python -m venv .venv

Activate it on Windows:

.venv\Scripts\Activate.ps1

Install the project and dependencies:

pip install -e .
pip install -r requirements.txt
Running the Project
Download the data
python scripts/download_data.py
Create the processed dataset and exploratory outputs
python scripts/make_features.py
Run benchmark models
python scripts/evaluate_models.py
Run SARIMA
python scripts/run_sarima.py
Run SARIMAX
python scripts/run_sarimax.py
Run Gradient Boosting
python scripts/run_feature_model.py
Run LSTM
python scripts/run_neural.py
Run the complete pipeline
python scripts/run_pipeline.py
Build the final model comparison only
python scripts/run_pipeline.py --comparison-only
Testing

Run the complete test suite:

pytest -v

The tests cover:

evaluation metrics;
benchmark forecasts;
leakage-safe lag and rolling features;
LSTM sequence creation;
pipeline behaviour.
Key Outputs
Final metrics
outputs/metrics/model_comparison.csv
Final comparison figure
outputs/figures/model_comparison.png
Model forecasts
outputs/forecasts/
Full report
reports/Electricity Demand Forecasting Report (24069723).pdf
Key Findings
Annual seasonality was the dominant forecasting signal.
Seasonal naïve was difficult to outperform.
Gradient Boosting produced the best numerical accuracy.
Its improvement over seasonal naïve was small.
SARIMA captured annual behaviour but retained shorter-term residual autocorrelation.
SARIMAX produced clean residuals but poor out-of-sample accuracy and high positive bias.
LSTM produced plausible forecasts but overfitted because of limited training sequences.
Greater model complexity did not automatically produce better forecasts.
Future Improvements

Possible extensions include:

rolling-origin evaluation;
weather forecasts instead of realised temperatures;
weather data from multiple German regions;
longer historical datasets;
hourly or daily modelling;
conformal or quantile prediction intervals for machine-learning models;
explicit treatment of structural breaks and unusual events.
Author

Sai Mahesh Battula
Student ID: 24069723

GitHub:
https://github.com/saimahesh1810/electricity-demand-forecasting