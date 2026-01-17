from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
import concurrent.futures


from src.agents import(
    generate_solution,
    refine_solution,
    review_solution,
    judge_debate,
    get_role_preference
)

# --- THE SHARED MEMORY STATE ---
class DebateState(TypedDict):
    problem_text: str

    #Store data from each stage
    roles: Dict[str, str]

    solvers_data: List[Dict]    # [{'agent': 'Solver1', 'solution': ...}, ...]
    reviews_data: List[Dict]    # [{'reviewer': 'Solver2', 'critique': ...}, ...]
    refined_data: List[Dict]    # [{'agent': 'Solver1', 'final': ...}, ...]

    final_verdict: Dict    # The Judge's output

# --- GRAPH NODES (The Steps) ---

def node_role_election(state: DebateState):
    """Judge Election Stage: 4 Candidates vote. Best Judge selected. Others become Solvers."""
    print("\n--- STAGE 0: ROLE ELECTION ---")
    problem = state["problem_text"]
    
    #Candidates will be popular AI's from the media i have consumed :DD 
    candidates = ["HAL 9000", "GLaDOS", "AM", "TARS"]
    
    def run_election_interview(agent_name):
        print(f"{agent_name} is deciding role preference...")
        pref = get_role_preference(problem, agent_name)
        return {
            "name": agent_name,
            "pref": pref.role_preference,
            "judge_score": pref.judge_confidence,
            "solver_score": pref.solver_confidence
        }

    votes = []
    # Run in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_election_interview, name) for name in candidates]
        for future in concurrent.futures.as_completed(futures):
            votes.append(future.result())   
    # Sort by 'judge_confidence' descending
    sorted_votes = sorted(votes, key=lambda x: x["judge_score"], reverse=True)
    
    judge = sorted_votes[0]["name"]
    solvers = [x["name"] for x in sorted_votes[1:]]
    
    print(f"ELECTED JUDGE: {judge} (Confidence: {sorted_votes[0]['judge_score']:.2f})")
    print(f"ASSIGNED SOLVERS: {solvers}")
    
    return {
        "roles": {
            "judge": judge,
            "solvers": solvers
        }
    }


def node_initial_solve(state: DebateState):
    """Stage 1: The 3 Elected Solvers attempt the problem."""
    print("\n--- STAGE 1: INDEPENDENT SOLVING ---")
    problem = state["problem_text"]
    my_solvers = state["roles"]["solvers"]
    
    def run_one_solver(agent_name):
        print(f"{agent_name} is thinking...")
        response = generate_solution(problem, agent_name)
        return {
            "agent": agent_name,
            "raw_response": response.dict(),
            "text": response.solution_text
        }

    results = []
    # Run in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(run_one_solver, name) for name in my_solvers]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    
    # Sort to maintain consistent order for indexing later
    results.sort(key=lambda x: x["agent"])
    
    return {"solvers_data": results}

def node_peer_review(state: DebateState):
    """Stage 2: Agents review each other's solutions."""
    print("\n--- STAGE 2: PEER REVIEW ---")
    problem = state["problem_text"]
    solvers = state["solvers_data"]
    
    tasks = []
    # Generate review pairs (Everyone reviews everyone else)
    for reviewer_idx in range(3):
        for target_idx in range(3):
            if reviewer_idx == target_idx: continue
            
            reviewer = solvers[reviewer_idx]["agent"]
            target_text = solvers[target_idx]["text"]
            target_name = solvers[target_idx]["agent"]
            tasks.append((reviewer, target_name, target_text))

    def run_one_review(task_data):
        r_name, t_name, t_text = task_data
        print(f"{r_name} is reviewing {t_name}...")
        critique = review_solution(problem, t_text)
        return {
            "reviewer": r_name,
            "target": t_name,
            "critique_object": critique.dict()
        }

    reviews = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(run_one_review, t) for t in tasks]
        for future in concurrent.futures.as_completed(futures):
            reviews.append(future.result())

    return {"reviews_data": reviews}

def node_refinement(state: DebateState):
    """Stage 3: Solvers fix their answers."""
    print("\n--- STAGE 3: REFINEMENT ---")
    problem = state["problem_text"]
    solvers = state["solvers_data"]
    reviews = state["reviews_data"]

    def run_one_refinement(i):
        agent_name = solvers[i]["agent"]
        original_text = solvers[i]["text"]
        
        # Gather all critiques for this agent
        my_critiques = [
            r["critique_object"]["weaknesses"] 
            for r in reviews 
            if r["target"] == agent_name
        ]
        
        combined_feedback = ""
        for idx, c in enumerate(my_critiques):
            combined_feedback += f"Critique {idx+1}: {str(c)}\n"
        
        print(f"{agent_name} is fixing their solution...")
        refined = refine_solution(problem, original_text, combined_feedback)
        
        return {
            "agent": agent_name,
            "final_text": refined.final_solution,
            "final_answer": refined.final_answer,
            "changes": refined.changes_made
        }

    refined_results = []
    
    # Run in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(run_one_refinement, i) for i in range(3)]
        for future in concurrent.futures.as_completed(futures):
            refined_results.append(future.result())
            
    refined_results.sort(key=lambda x: x["agent"])
        
    return {"refined_data": refined_results}

def node_judge(state: DebateState):
    """Stage 4: The Judge decides."""
    print("\n--- STAGE 4: FINAL JUDGMENT ---")
    problem = state["problem_text"]
    # We pass the refined solutions to the judge
    candidates = [r["final_text"] for r in state["refined_data"]]
    
    print("The Judge is choosing the best solution...")
    verdict = judge_debate(problem, candidates) # Returns Pydantic Object
    
    return {"final_verdict": verdict.dict()} 

# --- BUILD THE GRAPH ---

def build_debate_graph():
    builder = StateGraph(DebateState)

    #Add Nodes
    builder.add_node("election", node_role_election)
    builder.add_node("solve", node_initial_solve)
    builder.add_node("review", node_peer_review)
    builder.add_node("refine", node_refinement)
    builder.add_node("judge", node_judge)

    # Connect Edges
    builder.set_entry_point("election")
    builder.add_edge("election", "solve")
    builder.add_edge("solve", "review")
    builder.add_edge("review", "refine")
    builder.add_edge("refine", "judge")
    builder.add_edge("judge", END)

    return builder.compile()