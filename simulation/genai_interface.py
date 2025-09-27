import pandas as pd
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os
import streamlit as st


load_dotenv()

# Step 1: Set up AzureOpenAI API Key

# llm = AzureChatOpenAI(
#     api_key=os.getenv("AZURE_API_KEY"),
#     api_version=os.getenv("AZURE_API_VERSION"),
#     azure_endpoint=os.getenv("AZURE_ENDPOINT"),
#     deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"))
    # The response object has a `.content` property with the text

llm = AzureChatOpenAI(
    api_key=st.secrets["AZURE_API_KEY"],  # Fetch from Streamlit secrets
    api_version=st.secrets["AZURE_API_VERSION"],
    azure_endpoint=st.secrets["AZURE_ENDPOINT"],
    deployment_name=st.secrets["AZURE_DEPLOYMENT_NAME"]
)