import streamlit as st
from agent_core import create_agent


# --- Streamlit Configuration ---
st.set_page_config(page_title="LLM Tool", layout="wide")
API_KEY=st.secrets['DIAL_ACCESS_TOKEN']

# --- Google Gemini API Key ---
st.sidebar.header("Settings")
api_key = st.sidebar.text_input(API_KEY, type="password")

if not api_key:
    st.error("Please enter your API Key in the sidebar.")
    st.stop()

# --- Initialize LLM Agent ---
agent = create_agent(api_key)

st.title("Interactive LLM Tool")
st.write("Ask questions related to the dataset or request actionable insights!")

# --- User Input ---
query = st.text_input("Enter your query:")

if query:
    # Run the agent with the user query
    with st.spinner("Processing your query..."):
        try:
            response = agent.run(query)
            st.success("Response:")
            st.write(response)
        except Exception as e:
            st.error(f"Error during processing: {str(e)}")
else:
    st.info("Type a question in the text box above to get started!")
