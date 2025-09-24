import pandas as pd
import matplotlib.pyplot as plt

# Dataset as a CSV-like string
csv_path=r'C:\Users\jatin_arora\PycharmProjects\CapstoneProject\src\manufacturing_production_data.csv'
# Step 1: Load the dataset
def load_dataset(data):
    """Load dataset into DataFrame and preprocess."""
    df = pd.read_csv(data, parse_dates=["timestamp"])
    return df

df = load_dataset(csv_path)

# Step 2: Define Production Scenario Simulation
def simulate_production(df, production_rate_adjustment, max_temp_threshold, testing_defect_adjustment, shift_to_simulate):
    """
    Simulates different production scenarios based on:
      - Adjusting production rates.
      - Setting a temperature failure threshold.
      - Adjusting defect rates for the testing stage.
      - Simulating impact on a specific shift's output.
    """
    df_sim = df.copy()

    # Scenario 1: Adjust production rate across all shifts (output_qty + adjustment)
    df_sim['output_qty_simulated'] = df_sim['output_qty'] * (1 + production_rate_adjustment)

    # Scenario 2: Simulate temperature failures where `sensor_1_temp` exceeds the threshold
    df_sim['temp_failure_flag'] = df_sim['sensor_1_temp'] > max_temp_threshold
    df_sim.loc[df_sim['temp_failure_flag'], 'output_qty_simulated'] *= 0.8  # Reduce production by 20% on temp failure

    # Scenario 3: Adjust defect rate during the "testing" production stage
    df_sim.loc[df_sim['production_stage'] == 'testing', 'defect_flag_simulated'] = (
        df_sim['defect_flag'] + (df_sim['defect_flag'] * testing_defect_adjustment)
    )
    df_sim['defect_flag_simulated'] = df_sim['defect_flag_simulated'].fillna(0).astype(int)

    # Scenario 4: Focus on a specific shift and simulate its impact
    specific_shift_sim = df_sim[df_sim['shift'] == shift_to_simulate]

    return df_sim, specific_shift_sim

# Run the simulation
production_rate_adjustment = 0.1  # Increase production rate by 10%
max_temp_threshold = 75          # Set a high-temperature threshold
testing_defect_adjustment = 0.2  # Increase defect rate by 20% in testing stages
shift_to_simulate = 1            # Focus on Shift 1

simulated_df, shift_simulated_df = simulate_production(
    df, production_rate_adjustment, max_temp_threshold, testing_defect_adjustment, shift_to_simulate
)

# Step 3: Analyze Results
print("\n### SIMULATED DATA ###")
print(simulated_df)

print("\n### SIMULATED SHIFT SPECIFIC DATA ###")
print(shift_simulated_df)

# Step 4: Visualize the Simulation Results
def visualize_simulation(df_sim):
    """Visualize the Simulated Production Scenarios."""
    # Original vs Simulated Output Quantity
    plt.figure(figsize=(10, 6))
    plt.plot(df_sim['timestamp'], df_sim['output_qty'], label='Original Output Qty', marker='o')
    plt.plot(df_sim['timestamp'], df_sim['output_qty_simulated'], label='Simulated Output Qty', marker='x')
    plt.title('Original vs Simulated Production Output')
    plt.xlabel('Timestamp')
    plt.ylabel('Output Quantity')
    plt.legend()
    plt.grid()
    plt.show()

    # Defective Items in Testing Stage
    plt.figure(figsize=(8, 5))
    testing_stage = df_sim[df_sim['production_stage'] == 'testing']
    plt.bar(testing_stage['shift'], testing_stage['defect_flag_simulated'], color='orange', alpha=0.7, label='Simulated Defective Items')
    plt.title('Simulated Defective Items in Testing Stage (By Shift)')
    plt.xlabel('Shift')
    plt.ylabel('Defective Items')
    plt.legend()
    plt.show()

# Visualize the results
visualize_simulation(simulated_df)