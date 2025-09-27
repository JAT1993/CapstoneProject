import pandas as pd
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os
import streamlit as st


llm = AzureChatOpenAI(
    api_key=st.secrets["AZURE_API_KEY"],  # Fetch from Streamlit secrets
    api_version=st.secrets["AZURE_API_VERSION"],
    azure_endpoint=st.secrets["AZURE_ENDPOINT"],
    deployment_name=st.secrets["AZURE_DEPLOYMENT_NAME"]
)