import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path= 'manufacturing_production_data.csv'

# Step 1: Load the data into a Pandas DataFrame
# Import necessary libraries
import pandas as pd
import numpy as np


# --- STEP 1: LOAD THE CSV FILE ---
def load_csv(file_path):
    """
    Load the CSV file into a Pandas DataFrame.
    """
    try:
        # Parse the timestamp column
        df = pd.read_csv(file_path, parse_dates=["timestamp"])
        print("CSV file loaded successfully!")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at path '{file_path}'")
        return None
    except Exception as e:
        print(f"Error occurred while loading CSV: {e}")
        return None


# --- STEP 2: HANDLE MISSING VALUES ---
def handle_missing_data(df, drop_threshold=0.3):
    """
    Handle missing data:
    - Drop columns with % missing above the threshold.
    - Fill remaining missing values with appropriate strategies.
    """
    # Drop columns with excessive missing values (above threshold)
    missing_percentage = df.isnull().mean()
    cols_to_drop = missing_percentage[missing_percentage > drop_threshold].index
    df = df.drop(columns=cols_to_drop)
    print(f"Dropped columns with missing % above {drop_threshold * 100}: {list(cols_to_drop)}")

    # Fill numerical missing values with mean
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # Fill categorical missing values with mode
    categorical_cols = df.select_dtypes(include=["object"]).columns
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

    print("Handled missing values.")
    return df


# --- STEP 3: FEATURE ENGINEERING ---
def extract_features(df):
    """
    Extract features such as hour, day, and week from the timestamp column.
    """
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["week"] = df["timestamp"].dt.isocalendar().week
    df["month"] = df["timestamp"].dt.month
    df["year"] = df["timestamp"].dt.year
    print("Extracted timestamp features: hour, day, week, month, year.")

    # Example derived feature: temperature anomaly flag
    temp_threshold = df["sensor_1_temp"].mean() + (2 * df["sensor_1_temp"].std())
    df["temp_anomaly_flag"] = np.where(df["sensor_1_temp"] > temp_threshold, 1, 0)

    print("Derived the feature 'temp_anomaly_flag'.")
    return df


# --- STEP 4: PREPROCESS THE DATA ---
def preprocess_data(file_path):
    """
    Complete preprocessing pipeline:
    - Load CSV file.
    - Handle missing values.
    - Perform feature engineering.
    - Save the preprocessed dataset.
    """
    print("Beginning data ingestion and preprocessing...")
    df = load_csv(file_path)

    if df is None:
        return None  # Exit if loading failed

    df = handle_missing_data(df)
    df = extract_features(df)

    # Save the preprocessed data for downstream use
    preprocessed_file_path = "preprocessed_data.csv"
    df.to_csv(preprocessed_file_path, index=False)
    print(f"Preprocessed data saved to '{preprocessed_file_path}'.")
    return df


# Preprocess the data
df = preprocess_data(file_path)


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

# # Step 3: Visualize Data
# def visualize_data(df):
#     """
#     Create basic visualizations from manufacturing data.
#     """
#     # Sensor 1 temperature over time
#     plt.figure(figsize=(12, 6))
#     plt.plot(df['timestamp'], df['sensor_1_temp'], marker='o', label='Sensor 1 Temperature')
#     plt.title('Sensor 1 Temperature Over Time')
#     plt.xlabel('Timestamp')
#     plt.ylabel('Temperature')
#     plt.legend()
#     plt.grid()
#     plt.show()
#
#     # Sensor 2 vibration over time for each machine
#     plt.figure(figsize=(12, 6))
#     sns.lineplot(data=df, x='timestamp', y='sensor_2_vibration', hue='machine_id', marker='o')
#     plt.title('Sensor 2 Vibration Over Time (By Machine)')
#     plt.xlabel('Timestamp')
#     plt.ylabel('Vibration')
#     plt.legend(title='Machine ID')
#     plt.grid()
#     plt.show()
#
#     # Output quantity by shift
#     plt.figure(figsize=(8, 4))
#     sns.barplot(x='shift', y='output_qty', data=df, ci=None, estimator=sum, palette='Blues_d')
#     plt.title('Total Output Quantity by Shift')
#     plt.xlabel('Shift')
#     plt.ylabel('Total Output Quantity')
#     plt.grid()
#     plt.show()
#
#     # Defect flag distribution
#     plt.figure(figsize=(6, 4))
#     sns.countplot(data=df, x='defect_flag', palette='Set2')
#     plt.title('Defect Flag Distribution')
#     plt.xlabel('Defect Flag (0 = No Defect, 1 = Defect)')
#     plt.ylabel('Count')
#     plt.grid()
#     plt.show()
#
# # Perform visualizations
# visualize_data(df)