<p align="center">
  <img src="logo.png" width="170">
</p>

<h1 align="center">
Forecasting Train Passenger Volume in Indonesia using SARIMA
</h1>

<p align="center">
An interactive Streamlit application for forecasting monthly railway passenger volume in Indonesia using the <b>Seasonal Autoregressive Integrated Moving Average (SARIMA)</b> model.
</p>

<p align="center">

<a href="https://kai-passenger-forecasting-timeseries.streamlit.app/">
<img src="https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit-red?style=for-the-badge&logo=streamlit">
</a>

<img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python">

<img src="https://img.shields.io/badge/Model-SARIMA-success?style=for-the-badge">

<img src="https://img.shields.io/badge/Dataset-BPS-orange?style=for-the-badge">

</p>

---

## 📌 Project Overview

This project aims to forecast monthly railway passenger volume in Indonesia using historical data published by **Badan Pusat Statistik (BPS)**. The application is built with **Streamlit** and utilizes the **SARIMA** time series forecasting model to support transportation planning and operational decision-making.

### ✨ Features

- 📈 Forecast monthly railway passenger volume
- 🚉 Forecast by region:
  - National
  - Jabodetabek
  - Non-Jabodetabek
  - Java
  - Sumatra
- 📊 Interactive historical vs forecast visualization
- 📋 Forecast result table
- 📌 Forecast summary
- 📉 Model evaluation (MAE, RMSE, MAPE)
- 📱 Responsive interface for Desktop & Mobile

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Statsmodels (SARIMA)

---

## 📂 Dataset

**Source:** Badan Pusat Statistik (BPS) Indonesia

- Historical Period: **January 2006 – May 2026**
- Data Frequency: **Monthly**
- Unit: **Thousand Passengers**

> **Note**
>
> Regional passenger data are available only until **December 2023**. Therefore, the SARIMA model was trained using the complete national dataset, while the same modeling approach is consistently applied for regional forecasting visualization.

---

## 🤖 Forecasting Model

The following forecasting models were evaluated:

| Model | MAE | RMSE | MAPE |
|:----------------|---------:|---------:|---------:|
| Random Forest | 8,904.91 | 10,147.50 | **21.02%** |
| XGBoost | 12,002.28 | 13,997.90 | 27.93% |
| **SARIMA** | 17,373.61 | 19,145.57 | 42.60% |
| ARIMA | 18,410.88 | 20,149.45 | 45.34% |
| Holt-Winters | 20,101.08 | 22,193.01 | 49.22% |
| Prophet | 32,427.39 | 34,950.11 | 80.84% |

### Why SARIMA?

Although **Random Forest** achieved the lowest forecasting error, its long-term predictions were less realistic for time series behavior.

SARIMA was selected because it better preserves:

- Trend
- Seasonality
- Long-term forecasting stability

making it the most representative model for railway passenger forecasting.

---

## 📊 Dashboard Preview

- Forecast Configuration
- Historical vs Forecast Visualization
- Forecast Result Table
- Forecast Summary
- Model Evaluation

---

## 📦 Project Structure

```text
.
├── app.py
├── notebook.ipynb
├── streamlit_df.csv
├── logo.png
├── requirements.txt
├── README.md
└── ...
```

---

## ▶️ Installation

Clone the repository

```bash
git clone https://github.com/hastisf/kai-passenger-forecasting.git
```

Move into the project directory

```bash
cd kai-passenger-forecasting
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run Streamlit

```bash
streamlit run app.py
```

---

## 🌐 Live Demo

🚀 https://kai-passenger-forecasting-timeseries.streamlit.app/

---

## 👩‍💻 Author

**Hasti Sri Fatmawati**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Hasti%20Sri%20Fatmawati-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/hasti-sri-fatmawati-361b49417/)

[![GitHub](https://img.shields.io/badge/GitHub-hastisf-181717?style=flat-square&logo=github)](https://github.com/hastisf)

---

## 📄 License

This project was developed for educational purposes and portfolio demonstration.
