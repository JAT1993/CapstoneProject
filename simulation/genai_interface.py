import pandas as pd
from openai import AzureOpenAI
import streamlit as st

# Step 1: Set up AzureOpenAI API Key
API_KEY=st.secrets['DIAL_ACCESS_TOKEN']

client = AzureOpenAI(
    api_key=API_KEY,
    azure_endpoint="https://ai-proxy.lab.epam.com",
    api_version="2024-02-01",
    # azure_deployment="gpt-4o"
)

# Dataset as a CSV-like string
csv_path='src/manufacturing_production_data.csv'

# Step 2: Load the dataset into a Pandas DataFrame
def load_dataset(data):
    """Load dataset into DataFrame."""
    df = pd.read_csv(data, parse_dates=["timestamp"])
    return df

df = load_dataset(csv_path)

# Step 3: Create dataset summary functions
def summarize_dataset(df):
    """
    Calculate key metrics to generate insights.
    Returns a summary dictionary.
    """
    summary = {}
    summary['total_products'] = len(df)
    summary['defective_products'] = df['defect_flag'].sum()
    summary['defect_rate'] = (summary['defective_products'] / summary['total_products']) * 100
    summary['average_output'] = df['output_qty'].mean()
    summary['max_temp'] = df['sensor_1_temp'].max()
    summary['min_temp'] = df['sensor_1_temp'].min()
    summary['max_vibration'] = df['sensor_2_vibration'].max()
    summary['min_vibration'] = df['sensor_2_vibration'].min()
    summary['shift_output'] = df.groupby('shift')['output_qty'].sum().to_dict()

    return summary

summary = summarize_dataset(df)
print("### Summary ###")
print(summary)

# Step 4: Generate Prompt for GPT
def generate_prompt(summary):
    """
    Generate a dynamic prompt from the dataset summary.
    """
    prompt = f"""
    Here is the summary of the factory production data:
    - Total products produced: {summary['total_products']}
    - Defective products: {summary['defective_products']}
    - Defect rate: {summary['defect_rate']:.2f}%
    - Average output per production stage: {summary['average_output']:.2f}
    - Maximum temperature recorded by sensors: {summary['max_temp']:.2f}°C
    - Minimum temperature recorded by sensors: {summary['min_temp']:.2f}°C
    - Maximum vibration recorded by sensors: {summary['max_vibration']:.8f}
    - Minimum vibration recorded by sensors: {summary['min_vibration']:.8f}
    - Output by shift: {summary['shift_output']}.

    Based on these details, provide actionable insights and improvement recommendations
    for increasing production efficiency and reducing defects.
    """
    return prompt

prompt = generate_prompt(summary)

# Step 5: Use OpenAI API to Generate Insights
def generate_insights(prompt):
    """
    Use OpenAI GPT to generate insights based on the dataset summary.
    """
    try:
        # response = client.chat.completions.create(
        #     model="gpt-4o-mini-2024-07-18",  # You can also use GPT-3.5-Turbo or GPT-4
        #     prompt=prompt,
        #     max_tokens=300,  # Limit the number of tokens in the response
        #     temperature=0.7  # Adjust creativity level
        # )
        response = client.chat.completions.create(
			model="gpt-4o-mini-2024-07-18",
			messages=[{"role": "user", "content": prompt}]
		)

        # return response.choices[0].text.strip()
        return response


    except Exception as e:
        print(f"Error occurred during OpenAI API call: {e}")
        return None

# Get AI-generated insights
insights = generate_insights(prompt)
print("\n### AI-Generated Insights ###")
print(insights)