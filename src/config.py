import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

#load .env variables
load_dotenv()

#Configure constants
DATA_DIR = os.path.join(os.getcwd(), "data")
PROBLEMS_FILE = os.path.join(DATA_DIR, "problems.json")
RESULTS_DIR = os.path.join(DATA_DIR, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

def get_llm(temperature=0.7):
    """Returns the LLM instance"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API Key missing! Check your .env file.")
    
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=temperature,
        api_key=api_key
    )
    
