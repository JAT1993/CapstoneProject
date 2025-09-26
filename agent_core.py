import pandas as pd
from langchain.agents import initialize_agent, Tool, AgentType, AgentExecutor
from langchain.tools import tool
from langchain_community.chat_models import AzureChatOpenAI
from openai import AzureOpenAI
import streamlit as st
import os

# Load environment variables from .env (OpenAI API Key)
# load_dotenv()
API_KEY=st.secrets['DIAL_ACCESS_TOKEN']

# Dataset as a CSV-like string (mock data from the dataset you provided)


# Dataset as a CSV-like string
csv_path='src/manufacturing_production_data.csv'
client = AzureOpenAI(
    api_key=API_KEY,
    azure_endpoint="https://ai-proxy.lab.epam.com",
    api_version="2024-02-01",
    # azure_deployment="gpt-4o"
            )

# Step 1: Load the dataset into a Pandas DataFrame
def load_dataset(data):
    """Load dataset into DataFrame."""
    return pd.read_csv(data, parse_dates=["timestamp"])

df = load_dataset(csv_path)

# Step 2: Define a Tool for Dataset Exploration
@tool
def analyze_production_data(input_text: str) -> str:
    """
    A tool for analyzing the production dataset and answering questions about it.
    Input: Questions about production data.
    Output: Insights based on the dataset.
    """
    if "shift" in input_text.lower():
        shift_output = df.groupby("shift")["output_qty"].sum().to_string()
        return f"Output quantities per shift:\n{shift_output}"
    elif "sensor" in input_text.lower():
        sensor_summary = {
            "Max Sensor 1 Temp": df["sensor_1_temp"].max(),
            "Min Sensor 1 Temp": df["sensor_1_temp"].min(),
            "Max Sensor 2 Vibration": df["sensor_2_vibration"].max(),
            "Min Sensor 2 Vibration": df["sensor_2_vibration"].min()
        }
        return f"Sensor summary:\n{sensor_summary}"
    elif "defect" in input_text.lower():
        defect_rate = (df["defect_flag"].sum() / len(df)) * 100
        return f"Defect rate is {defect_rate:.2f}%."
    else:
        return "I couldn't understand your question. Please ask about shifts, sensors, or defects."

# Step 3: Define a Tool for Generating Insights Using OpenAI GPT
@tool
def generate_insights(input_text: str) -> str:
    """
    Uses GPT to generate actionable insights.
    """
    try:
        # Initialize OpenAI Chat model
        # chat_model = ChatOpenAI(temperature=0.7, openai_api_key=openai_api_key)


        # Use ChatOpenAI to generate a response
        # response = chat_model(input_text)
        response = client.chat.completions.create(
			model="gpt-4o-mini-2024-07-18",
			messages=[{"role": "user", "content": input_text}]
		)
        return response
    except Exception as e:
        return f"Error generating insights: {str(e)}"

# Step 4: Initialize LangChain Tools
tools = [
    Tool(name="Data Analysis Tool", func=analyze_production_data, description="Analyze production data."),
    Tool(name="Insight Generation Tool", func=generate_insights, description="Generate actionable insights using GPT.")
]

# Step 5: Create LangChain Agent
def create_agent(tools):
    """
    Initializes LangChain Agent framework with tools.
    """
    from langchain.chat_models import ChatOpenAI

    chat_model = AzureChatOpenAI(
    openai_api_base="https://ai-proxy.lab.epam.com",  # Replace with your Azure endpoint
    openai_api_key=API_KEY,                         # Azure OpenAI API Key
    deployment_name="gpt-4o-mini-2024-07-18",                           # Model deployment name in Azure
    openai_api_version="2024-02-01",                             # Azure API Version (verify in Azure portal)
    temperature=0.7                                              # Adjust creativity level
)

    # response = chat_model.predict("Write me a story about the ocean.")
    # print(response)
    # chat_model = ChatOpenAI(temperature=0.7, openai_api_key=openai_api_key)
    agent = initialize_agent(tools, chat_model, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    return agent

agent = create_agent(tools)

# Step 6: Run the Agent with Prompts
print("\n### Running Agent ###")
query_1 = "What's the output quantity per shift?"
response_1 = agent.invoke(query_1)
print(f"\nAgent Response for query1:\n{response_1}")

query_2 = "What are the sensor readings?"
response_2 = agent.invoke(query_2)
print(f"\nAgent Response for query2:\n{response_2}")

query_3 = "Can you generate insights for reducing defects based on the data?"
response_3 = agent.invoke(query_3)
print(f"\nAgent Response for query3:\n{response_3}")