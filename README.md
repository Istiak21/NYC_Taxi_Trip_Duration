# 🚖 NYC Taxi Trip Duration Prediction

This project focuses on predicting the duration of NYC taxi trips using various machine learning models. The dataset contains trip records including pickup/dropoff datetime and coordinates. The goal is to predict the trip duration using both spatial and temporal features.

---

## 📊 Project Highlights

- Preprocessing of pickup and dropoff datetime
- Feature engineering with time, location, distance, and direction features
- Application of haversine formula and bearing calculation for spatial features
- Implemented regression models: **Linear Regression**, **Decision Tree**
- Evaluation of models using **R² Score**

---

## 📁 Dataset

The dataset is loaded from a ZIP file named `nyc_taxi_final.zip`, which contains records of taxi trips in New York City. Each record includes:

- Pickup and dropoff timestamps
- Pickup and dropoff coordinates
- Trip duration (in seconds)
- Vendor ID and other metadata

---

## 🧠 Features Engineered

- **Temporal Features**: Weekday, Week of Year, Hour, Minute
- **Spatial Features**: Haversine Distance, Direction (Bearing), Rounded Coordinates
- **Target Variable**: Log-transformed trip duration using `np.log1p(trip_duration)`

---

## 🧮 Algorithms Used

- 📈 **Linear Regression**
- 🌳 **Decision Tree Regression**

---

## 📐 Metrics

Model performance was evaluated using:

- **R² Score**
- **Root Mean Squared Error (RMSE)**

## 📋 Results

```text
R² Score for Linear Regression: 0.3631
R² Score for Decision Tree:    0.7010
