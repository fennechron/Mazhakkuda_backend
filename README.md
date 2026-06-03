# AI-Powered Travel Weather Assistant (Destination-Based Rain Predictor)

A self-learning machine learning system that predicts whether it will rain at your destination at your estimated time of arrival, rather than predicting the current weather at your starting location.

The system integrates routing intelligence (OpenStreetMap Nominatim and OSRM driving route engine) with meteorological forecasts (OpenWeatherMap) and a Random Forest Classifier that corrects forecast errors based on historical region-specific patterns and real-time user feedback.

---

## 📂 Project Structure

```
rain_prediction_model/
│
├── dataset/
│   └── weather_data.csv            # Travel weather database (features, predictions, actual feedback)
│
├── model/
│   └── rain_model.pkl              # Serialized RandomForest classifier
│
├── route_service.py                # Geocodes cities (Nominatim) and calculates driving routes (OSRM)
├── travel_time_estimator.py        # Coordinates travel duration calculation with simulation fallback
├── destination_forecast.py         # Matches future destination forecast to estimated arrival time
│
├── train_model.py                  # Trains the Random Forest model on travel features
├── predict_model.py                # Computes travel routes, forecasts, and runs AI prediction
├── preprocess.py                   # Imputes, cleans, and encodes location strings deterministically
├── save_feedback.py                # Portal for logging ground-truth feedback at the destination
├── retrain_model.py                # Automated continuous learning and candidate deployment pipeline
├── utils.py                        # Common operations, dataset I/O, and OWM API accuracy metrics
│
├── requirements.txt                # Python package list
└── README.md                       # Project documentation (this file)
```

---

## ⚡ Quick Start

### 1. Installation
Ensure you have Python 3 installed. Navigate to the project directory and install the requirements:
```bash
pip install -r requirements.txt
```

### 2. Configure Weather API (Optional but Recommended)
Create a `.env` file in the root folder (`rain_prediction_model/`) to enable live forecast queries:
```env
OPENWEATHERMAP_API_KEY=your_api_key_here
```
*Note: If no API key is specified, the system will automatically fall back to simulated weather forecast generation, allowing you to test the complete pipeline offline.*

---

## 🛠️ How to Run the System

The pipeline operates in a continuous self-learning loop:
`Seed Data` ➡️ `Train Initial Model` ➡️ `Predict Travel Route Weather` ➡️ `Submit Arrival Feedback` ➡️ `Retrain Model` ➡️ `Repeat`.

### Step 1: Train the Travel Prediction Model
The dataset is seeded with 100 historical travel scenarios. Run the initial training script:
```bash
python train_model.py
```
This output includes accuracy, precision, recall, and a confusion matrix evaluated on a 20% test split.

### Step 2: Make a Travel Prediction
Calculate driving duration, estimate arrival time, and predict whether it will rain at the destination when you arrive.

* **Automatic Mode (Live Nominatim + OSRM + OWM)**:
  Queries real travel distances and pulls the destination forecast aligned to the arrival time.
  ```bash
  python predict_model.py --start "Kumbanad" --dest "Chengannur"
  ```
* **Specify Manual Duration**:
  Override the route estimator with a manual travel duration (in minutes):
  ```bash
  python predict_model.py --start "Kumbanad" --dest "Chengannur" --duration 45
  ```
* **Offline Simulation Mode**:
  Forces the system to use simulated routing metrics and simulated forecast grids:
  ```bash
  python predict_model.py --start "Kumbanad" --dest "Chengannur" --simulate
  ```
* **Manual Feature Entry**:
  Type custom meteorological numbers directly in the terminal to test hypothetical travel scenarios:
  ```bash
  python predict_model.py --start "Kumbanad" --dest "Chengannur" --manual
  ```

This outputs a **🔔 Travel Notification Preview** (e.g. `☔ Rain expected in Chengannur around 05:15 PM` or `☀ No rain expected during your travel`) and logs the run to the dataset as a pending record.

### Step 3: Log Feedback on Arrival
When you reach your destination, record the actual weather outcome:
* **Interactive Mode**:
  ```bash
  python save_feedback.py
  ```
* **Command-line flags**:
  ```bash
  python save_feedback.py --rain      # If it rained at your destination at arrival time
  python save_feedback.py --no-rain   # If it did not rain at your destination at arrival time
  ```
The portal logs the feedback and calculates the prediction correctness.

### Step 4: Retrain and Self-Correct
Trigger retraining to update the model weights using the new feedback:
```bash
python retrain_model.py
```
The script evaluates the updated data, compares the new candidate classifier against the active baseline, performs a safety threshold check, and replaces the model on disk if the candidate is stable.

---

## 🧠 Travel Feature Engineering

The Machine Learning model operates on the following feature vector:
1. `current_location`: Deterministic hash representation of the start city.
2. `destination_location`: Deterministic hash representation of the target city.
3. `travel_duration`: Route driving duration in minutes.
4. `destination_temperature`: Temperature forecasted at target arrival time.
5. `destination_humidity`: Humidity forecasted at target arrival time.
6. `destination_pressure`: Atmospheric pressure forecasted at target arrival time.
7. `destination_wind_speed`: Wind speed forecasted at target arrival time.
8. `destination_cloud_coverage`: Cloud coverage percentage forecasted at target arrival time.
9. `destination_api_rain_probability`: Probability of precipitation forecasted at target arrival time.
10. `time_of_day`: Arrival hour (0–23).
11. `month`: Arrival month (1–12).
12. `historical_prediction_accuracy`: A rolling metric of the OWM API's accuracy for that specific destination city (or overall dataset accuracy if city-specific records are scarce).
