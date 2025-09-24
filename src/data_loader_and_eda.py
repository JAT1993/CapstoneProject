import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

csv_path='src/manufacturing_production_data.csv'

# Step 1: Load the data into a Pandas DataFrame
def load_data(data_source: str):
    """Loads the mock manufacturing data into a DataFrame."""
    df = pd.read_csv(data_source)
    df['timestamp'] = pd.to_datetime(df['timestamp'])  # Convert timestamp column to datetime
    return df

# Load the data

df = load_data(csv_path)

# Step 2: Perform Exploratory Data Analysis (EDA)
def eda(df):
    """Perform basic EDA on the manufacturing data."""
    print("### BASIC INFORMATION ###")
    print(df.info())  # Overview of the dataset

    print("\n### MISSING VALUES ###")
    print(df.isnull().sum())  # Check for missing values

    print("\n### SUMMARY STATISTICS ###")
    print(df.describe())  # Summary statistics for numerical columns

    print("\n### CATEGORICAL VARIABLES ###")
    print("Unique values per categorical column:")
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        print(f"{col}: {df[col].nunique()} unique values")
        print(df[col].value_counts())

    print("\n### SAMPLE DATA ###")
    print(df.head())  # Display the first 5 rows of the dataset

# Perform EDA
print("Perform EDA")
eda(df)

# Step 3: Visualize Data
def visualize_data(df):
    """
    Create basic visualizations from manufacturing data.
    """
    # Sensor 1 temperature over time
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['sensor_1_temp'], marker='o', label='Sensor 1 Temperature')
    plt.title('Sensor 1 Temperature Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Temperature')
    plt.legend()
    plt.grid()
    plt.show()

    # Sensor 2 vibration over time for each machine
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x='timestamp', y='sensor_2_vibration', hue='machine_id', marker='o')
    plt.title('Sensor 2 Vibration Over Time (By Machine)')
    plt.xlabel('Timestamp')
    plt.ylabel('Vibration')
    plt.legend(title='Machine ID')
    plt.grid()
    plt.show()

    # Output quantity by shift
    plt.figure(figsize=(8, 4))
    sns.barplot(x='shift', y='output_qty', data=df, ci=None, estimator=sum, palette='Blues_d')
    plt.title('Total Output Quantity by Shift')
    plt.xlabel('Shift')
    plt.ylabel('Total Output Quantity')
    plt.grid()
    plt.show()

    # Defect flag distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='defect_flag', palette='Set2')
    plt.title('Defect Flag Distribution')
    plt.xlabel('Defect Flag (0 = No Defect, 1 = Defect)')
    plt.ylabel('Count')
    plt.grid()
    plt.show()

# Perform visualizations
visualize_data(df)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset as a CSV-like string
csv_path='src/manufacturing_production_data.csv'

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
        # Z-Score-based anomaly simulation
        mean_val = df[feature].mean()
        std_dev = df[feature].std()
        df['z_score'] = (df[feature] - mean_val) / std_dev
        anomalies = df[abs(df['z_score']) > threshold]
    elif method == "iqr":
        # IQR-based anomaly simulation
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        anomalies = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
    else:
        raise ValueError("Unsupported method. Use 'zscore' or 'iqr'.")

    return anomalies

# Anomaly simulation for sensor readings
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