import json
import os 
from typing import List, Dict 
from src.config import PROBLEMS_FILE

def load_problems() -> List[Dict]:
    """
    Reads the dataset from data/problems.json.
    Returns a list of problem dictionaries.
    """
    if not os.path.exists(PROBLEMS_FILE):
        raise FileNotFoundError(f" Could not find dataset at: {PROBLEMS_FILE}")
    
    try:
        with open(PROBLEMS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        print(f"Successfully loaded {len(data)} problems from {PROBLEMS_FILE}")
        return data
    
    except json.JSONDecodeError:
        raise ValueError(f" Error decoding JSON in {PROBLEMS_FILE}. Check formatting.")
    except Exception as e:
        raise RuntimeError(f" Unexpected error loading data: {str(e)}")
    
# Test function to verify it works when run directly
if __name__ == "__main__":
    problems = load_problems()
    print(f"Sample Problem 1: {problems[0]['question']}")
    print(f"Sample Problem 2: {problems[1]['question']}")