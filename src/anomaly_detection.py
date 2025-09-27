import pandas as pd
from langchain.tools import tool

# Dataset as a CSV-like string
csv_path='data/preprocessed_data.csv'
# csv_path=r'C:\Users\jatin_arora\PycharmProjects\CapstoneProject\data\preprocessed_data.csv'
df = pd.read_csv(csv_path)

# Tool 2: Define Quality Issues and Metrics
@tool
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



# Tool 3: Define Anomaly Detection Methods
@tool
def detect_anomalies_zscore(df: pd.DataFrame, feature: str, threshold: float = 3.0) -> pd.DataFrame:
    """
    Detect anomalies in a specific numeric feature using Z-Score method.
    Inputs:
    - df: Pandas DataFrame.
    - feature: The feature/column name to analyze.
    - threshold: Z-Score threshold for detecting anomalies.

    Outputs:
    - DataFrame of detected anomalies.
    """
    print(f"\n### DETECTING ANOMALIES USING Z-SCORE IN FEATURE '{feature}' ###")

    # Calculate Z-Score
    mean_val = df[feature].mean()
    std_dev = df[feature].std()
    df['z_score'] = (df[feature] - mean_val) / std_dev  # Add Z-Score column
    anomalies = df[abs(df['z_score']) > threshold]

    return anomalies


# --- Tool 4: IQR Anomaly Detection ---
@tool
def detect_anomalies_iqr(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """
    Detect anomalies in a specific numeric feature using IQR (Interquartile Range) method.
    Inputs:
    - df: Pandas DataFrame.
    - feature: The feature/column name to analyze.

    Outputs:
    - DataFrame of detected anomalies.
    """
    print(f"\n### DETECTING ANOMALIES USING IQR IN FEATURE '{feature}' ###")

    # Calculate IQR
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    anomalies = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
    return anomalies