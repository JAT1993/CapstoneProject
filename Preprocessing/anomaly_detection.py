import pandas as pd
import matplotlib.pyplot as plt

# Dataset as a CSV-like string
csv_path=r'C:\Users\jatin_arora\PycharmProjects\CapstoneProject\src\manufacturing_production_data.csv'

# Step 1: Load the dataset
def load_dataset(data):
    """Load dataset into DataFrame and preprocess."""
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    return df

df = load_dataset(csv_path)

# Step 2: Define Quality Issues and Metrics
def quality_analysis(df):
    """Perform quality checks."""
    print("### PRODUCT QUALITY ANALYSIS ###")
    # Count defective products
    defect_counts = df['defect_flag'].value_counts()
    total_products = len(df)
    defect_rate = (defect_counts.get(1, 0) / total_products) * 100  # % of defective products

    print(f"Total Products: {total_products}")
    print(f"Defective Products: {defect_counts.get(1, 0)} ({defect_rate:.2f}%)")

    # Return defective rows for further inspection
    defective_rows = df[df['defect_flag'] == 1]
    return defective_rows

defective_products = quality_analysis(df)

# Step 3: Define Anomaly Detection Methods
def detect_anomalies(df, feature, method="zscore", threshold=3.0):
    """
    Detect anomalies in a specific numeric feature using provided method.
    - method: 'zscore' or 'iqr'.
    - threshold: For z-score, values beyond this threshold are flagged.
    """
    if method == "zscore":
        # Z-Score-based anomaly detection
        mean_val = df[feature].mean()
        std_dev = df[feature].std()
        df['z_score'] = (df[feature] - mean_val) / std_dev
        anomalies = df[abs(df['z_score']) > threshold]
    elif method == "iqr":
        # IQR-based anomaly detection
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        anomalies = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
    else:
        raise ValueError("Unsupported method. Use 'zscore' or 'iqr'.")

    return anomalies

# Anomaly detection for sensor readings
temp_anomalies = detect_anomalies(df, feature="sensor_1_temp", method="zscore")
vibration_anomalies = detect_anomalies(df, feature="sensor_2_vibration", method="iqr")

print("\n### TEMPERATURE ANOMALIES ###")
print(temp_anomalies[['timestamp', 'machine_id', 'sensor_1_temp', 'z_score']])

print("\n### VIBRATION ANOMALIES ###")
print(vibration_anomalies[['timestamp', 'machine_id', 'sensor_2_vibration']])

# Step 4: Visualize Results
def visualize_anomalies(df, temp_anomalies, vibration_anomalies):
    """Plot data with anomalies highlighted."""
    plt.figure(figsize=(10, 6))

    # Sensor 1 Temperature
    plt.plot(df['timestamp'], df['sensor_1_temp'], label='Sensor 1 Temperature', marker='o')
    plt.scatter(temp_anomalies['timestamp'], temp_anomalies['sensor_1_temp'], color='red', label='Temp Anomalies')
    plt.xlabel('Timestamp')
    plt.ylabel('Temperature')
    plt.title('Sensor 1 Temperature with Anomalies')
    plt.legend()
    plt.grid()
    plt.show()

    # Sensor 2 Vibration
    plt.figure(figsize=(10, 6))
    plt.plot(df['timestamp'], df['sensor_2_vibration'], label='Sensor 2 Vibration', marker='o')
    plt.scatter(vibration_anomalies['timestamp'], vibration_anomalies['sensor_2_vibration'], color='red', label='Vibration Anomalies')
    plt.xlabel('Timestamp')
    plt.ylabel('Vibration')
    plt.title('Sensor 2 Vibration with Anomalies')
    plt.legend()
    plt.grid()
    plt.show()

visualize_anomalies(df, temp_anomalies, vibration_anomalies)