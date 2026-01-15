import json
import os
import time
from tqdm import tqdm
from langchain_openai import ChatOpenAI
from collections import Counter

from src.dataloader import load_problems
from src.workflow.graph import build_debate_graph
from src.config import RESULTS_DIR, get_llm

def run_baseline_gpt4(question):
    """
    Runs a single call to GPT-4o-mini to act as the Baseline.
    We compare our debate system against this.
    """
    llm = get_llm(temperature=0)
    try:
        response = llm.invoke(f"Solve this problem concisely: {question}")
        return response.content
    except Exception as e:
        return f"Error: {e}"
    
def get_voting_baseline(solvers_data):
    """
    Extracts the short answers from the 3 initial solvers and finds the majority.
    """    
    answers = [s['raw_response']['final_answer_short'] for s in solvers_data]

    #Count frequnecies
    counts = Counter(answers)
    most_common = counts.most_common(1)

    winner, count = most_common[0]

    if count >= 2:
        return winner # Majority found
    else:
        return answers[0] # No consensus. Pick Solver 1
    
def main():
    print("Starting Multi-LLM Debate System")   

    # 1. Load Data
    problems = load_problems()
    debate_graph = build_debate_graph()

    results_log = []

    print(f"Found{len(problems)} problems. Starting processing.") 

    # 2. Loop through problems
    for problem in tqdm(problems):
        p_id = problem['id']
        question = problem['question']
        correct = problem['correct_answer']

        print(f"\n\n Processing Problem {p_id}: {question[:50]}...")

        # --- A. Run Debate System
        start_time = time.time()
        initial_state = {
            "problem_text": question,
            "solvers_data": [],
            "reviews_data": [],
            "refined_data": [],
            "final_verdict": {}
        }

        #Invoke LangGraph
        final_state = debate_graph.invoke(initial_state)
        debate_time = time.time() - start_time

        verdict = final_state["final_verdict"]
        winner = verdict.get("winner_id", "Unknown")
        final_ans = verdict.get("final_answer_text", "No Answer")

        print(f"Judge's Verdict: {winner}")
        print(f"Final Answer: {final_ans}")

        # --- B. Run Baseline
        print("Running Baseline (Single LLM)")
        baseline_ans = run_baseline_gpt4(question)

        # --- C. SAVE RESULTS ---
        log_entry = {
            "id": p_id,
            "category": problem['category'],
            "difficulty": problem['difficulty'],
            "question": question,
            "correct_answer": correct,
            "debate_answer": final_ans,
            "baseline_answer": baseline_ans,
            "judge_reasoning": verdict.get("reasoning", ""),
            "time_taken": debate_time,
            "full_state": final_state # Saves the whole debate history
        }
        results_log.append(log_entry)

        # Save incrementally (so we don't lose data if it crashes)
        output_file = os.path.join(RESULTS_DIR, "debate_results.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results_log, f, indent=2)

    print(f"\n Operation Finished. Results saved to {RESULTS_DIR}/debate_results.json")

if __name__ == "__main__":
    main()