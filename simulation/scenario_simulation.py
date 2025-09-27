import pandas as pd
from langchain.tools import tool

file_path='data/preprocessed_data.csv'
# file_path=r'C:\Users\jatin_arora\PycharmProjects\CapstoneProject\data\preprocessed_data.csv'

# --- Tool 1: Load Dataset ---
@tool
def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Load dataset into a Pandas DataFrame and preprocess timestamps.
    Inputs:
    - file_path: Path to the CSV file.

    Outputs:
    - DataFrame containing the loaded dataset.
    """
    try:
        df = pd.read_csv(file_path, parse_dates=["timestamp"])
        print("Dataset loaded successfully!")
        return df
    except Exception as e:
        raise ValueError(f"Error loading dataset: {str(e)}")


# --- Tool 2: Simulate Production Adjustment ---
@tool
def simulate_production(
        df: pd.DataFrame,
        production_rate_adjustment: float = 0.0,
        max_temp_threshold: float = 80.0,
        testing_defect_adjustment: float = 0.0,
        shift_to_simulate: int = None,
        apply_maintenance_downtime: bool = False,
        maintenance_downtime_shift: int = 2
) -> dict:
    """
    Simulates different production scenarios based on:
      - Adjusting production rates.
      - Simulating high-temperature sensor failures and reducing output.
      - Adding defect rate adjustments in the "testing" stage.
      - Simulating maintenance downtime on specific shifts.
      - Option to focus on a specific shift's impact.

    Inputs:
    - df (DataFrame): Original production dataset.
    - production_rate_adjustment (float): Production rate adjustment multiplier (+ve or -ve).
    - max_temp_threshold (float): High-temperature threshold to trigger output reduction.
    - testing_defect_adjustment (float): Adjustment factor for defect rates in the "testing" stage.
    - shift_to_simulate (int, optional): Shift number to analyze separately.
    - apply_maintenance_downtime (bool): If True, simulate downtime for a specific shift.
    - maintenance_downtime_shift (int): Shift affected by maintenance downtime.

    Outputs:
    - Dictionary with two keys:
      - 'simulated_df': Full dataset with simulations applied.
      - 'specific_shift_sim': Shift-specific results with applied simulations.
    """
    df_sim = df.copy()

    # --- Scenario 1: Adjust Production Rates ---
    df_sim['output_qty_simulated'] = df_sim['output_qty'] * (1 + production_rate_adjustment)
    print(f"Simulation: Adjusted production rate by {production_rate_adjustment * 100:.1f}%")

    # --- Scenario 2: Simulate Temperature Failures ---
    df_sim['temp_failure_flag'] = df_sim['sensor_1_temp'] > max_temp_threshold
    df_sim.loc[df_sim['temp_failure_flag'], 'output_qty_simulated'] *= 0.8  # Reduce production by 20% on failure
    temp_failures = df_sim['temp_failure_flag'].sum()
    print(f"Simulation: Applied temperature failure threshold ({max_temp_threshold}) - {temp_failures} rows impacted.")

    # --- Scenario 3: Adjust Defect Rates in Testing Stage ---
    df_sim['defect_flag_simulated'] = df_sim['defect_flag']  # Carry original defect flag
    df_sim.loc[df_sim['production_stage'] == 'testing', 'defect_flag_simulated'] = (
            df_sim['defect_flag'] * (1 + testing_defect_adjustment)
    ).fillna(0).astype(int)
    print(f"Simulation: Adjusted defect rates in 'testing' stage by {testing_defect_adjustment * 100:.1f}%")

    # --- Scenario 4: Maintenance Downtime (Optional) ---
    if apply_maintenance_downtime:
        # Apply maintenance downtime: Reduce output by 50% for the selected shift
        df_sim.loc[df_sim['shift'] == maintenance_downtime_shift, 'output_qty_simulated'] *= 0.5
        print(
            f"Simulation: Applied maintenance downtime for shift {maintenance_downtime_shift} (50% output reduction).")

    # --- Scenario 5: Focus on a Specific Shift ---
    specific_shift_sim = None
    if shift_to_simulate is not None:
        specific_shift_sim = df_sim[df_sim['shift'] == shift_to_simulate]
        print(f"Simulation: Data filtered for shift {shift_to_simulate}.")

    return {
        "simulated_df": df_sim,
        "specific_shift_sim": specific_shift_sim
    }