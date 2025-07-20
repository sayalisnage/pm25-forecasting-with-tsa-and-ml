# PM2.5 Air Quality Forecasting: A Hybrid Modeling Approach

A data science project focused on predicting hourly PM2.5 air pollution levels in Los Angeles. This project leverages historical air quality data and meteorological features using a hybrid approach combining time series and machine learning models to provide accurate, short-term forecasts.

---

## Project Overview

Air pollution, particularly fine particulate matter (PM2.5), poses significant public health risks in urban environments. This project aims to develop a robust forecasting system that provides localized, short-term PM2.5 predictions. This capability can inform public health decisions, guide environmental policy responses, and support proactive air quality management by individuals and authorities.

---

## Data Sources

* **OpenAQ API:** Provides historical hourly PM2.5 concentration data, serving as the primary target variable.
    * Source: [https://openaq.org/](https://openaq.org/)
* **OpenWeatherMap API:** Supplies comprehensive hourly meteorological variables (e.g., temperature, humidity, wind speed, pressure, rainfall) used as exogenous features.
    * Source: [https://openweathermap.org/api](https://openweathermap.org/api)

---

## Methodology

* **Data Acquisition & Preprocessing:** Hourly PM2.5 and weather data were acquired from open APIs, followed by extensive cleaning, time synchronization, and initial feature engineering. This prepares a unified dataset for in-depth analysis.
* **Exploratory Data Analysis (EDA):** In-depth analysis conducted in a Jupyter notebook to uncover temporal patterns (diurnal, weekly, seasonal) and relationships between PM2.5 and meteorological variables.
* **Feature Engineering:** Further creation of crucial features including lagged PM2.5 values, rolling means, and various temporal indicators (e.g., hour, day of week, month, daylight flags) within the analytical notebook.
* **Predictive Modeling:** Implementation and evaluation of a hybrid approach utilizing:
    * **Classical Time Series Models:** SARIMA and SARIMAX.
    * **Machine Learning Models:** Random Forest Regressor and XGBoost Regressor.
* **Model Evaluation:** Performance assessed using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Symmetric Mean Absolute Percentage Error (SMAPE).

---

## Technology Stack

* **Programming Language:** Python
* **Key Libraries & Tools:**
    * **Data Manipulation:** Pandas, NumPy
    * **Time Series Modeling:** `statsmodels` (for SARIMA, SARIMAX)
    * **Machine Learning:** `scikit-learn`, `xgboost`
    * **Data Visualization:** Matplotlib, Seaborn
    * **Development:** Jupyter Notebooks, Git

---

## How to Run the Project

To set up and run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[your-github-username]/pm25-air-quality-forecasting.git
    cd pm25-air-quality-forecasting
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Acquire Raw Data:**
    * Obtain API keys for OpenAQ and OpenWeatherMap.
    * Place your API keys in a `.env` file or directly within the data acquisition scripts (refer to script comments for details).
    * Run the data acquisition scripts to fetch raw data:
        ```bash
        python scripts/data_acquisition/oaq_data_acquisition.py
        python scripts/data_acquisition/owm_data_acquisition.py
        ```
5.  **Preprocess and Combine Data:**
    * Run the preprocessing script to clean and combine the raw data into a unified file for further analysis:
        ```bash
        python scripts/data_preprocessing.py
        ```
    * This script will save the processed data into the `data/processed/` directory.
6.  **Explore, Model, and Evaluate in Jupyter:**
    * Launch Jupyter Notebook or JupyterLab from the project root:
        ```bash
        jupyter notebook
        ```
    * Navigate to the `notebooks/` directory and open `pm25_forecasting_analysis.ipynb`.
    * Execute the cells in the notebook sequentially to perform EDA, feature engineering, model training, and evaluation.

---

## Results

* Achieved reliable short-term prediction accuracy for hourly PM2.5 concentrations in Los Angeles.
* The **Random Forest Regressor** consistently demonstrated superior performance (MAE: 2.12, RMSE: 3.03, SMAPE: 17.09%), significantly outperforming traditional time series models.
* The project successfully identified and leveraged key meteorological variables (e.g., wind speed, rainfall) and temporal patterns as influential factors for PM2.5 levels.

---

## Future Improvements

* **Advanced Modeling:** Explore deep learning architectures (e.g., LSTMs, Transformers) for enhanced sequence modeling capabilities.
* **External Factors:** Integrate additional contextual data such as traffic volume, industrial activity, or wildfire smoke data.
* **Multi-Location Forecasting:** Expand the project to forecast air quality across multiple cities or regions.
* **Real-time Deployment:** Develop a user-friendly web interface or API for real-time interactive forecasts.
* **Uncertainty Quantification:** Implement methods to provide forecast intervals and confidence levels.

## Contact

For questions or collaboration, please reach out via [E-mail](sayalinage@gmail.com) or connect on [LinkedIn](https://www.linkedin.com/in/sayali-nage-34303b136/).

---
