from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END


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
    votes = []
    
    # 1. Ask everyone
    for agent in candidates:
        print(f"{agent} is deciding role preference...")
        pref = get_role_preference(problem, agent)
        votes.append({
            "name": agent,
            "pref": pref.role_preference,
            "judge_score": pref.judge_confidence,
            "solver_score": pref.solver_confidence
        })
        
    # 2. Deterministic Algorithm (Stage 0.5)
    # Sort by 'judge_confidence' descending. Top 1 is Judge.
    sorted_votes = sorted(votes, key=lambda x: x["judge_score"], reverse=True)
    
    judge = sorted_votes[0]["name"]
    solvers = [x["name"] for x in sorted_votes[1:]] # The other 3
    
    print(f"ELECTED JUDGE: {judge} (Confidence: {sorted_votes[0]['judge_score']})")
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
    results = []    

    for agent_name in my_solvers:
        print(f" {agent_name} is thinking...")
        response = generate_solution(problem, agent_name)
        
        results.append({
            "agent": agent_name,
            "raw_response": response.dict(),
            "text": response.solution_text
        })
        
    return {"solvers_data": results}

def node_peer_review(state: DebateState):
    """Stage 2: Agents review each other's solutions."""
    print("\n--- STAGE 2: PEER REVIEW ---")
    problem = state["problem_text"]
    solvers = state["solvers_data"]
    reviews = []
    
    pairs = [(0, 1), (1, 2), (2, 0)] 
    
    for reviewer_idx, target_idx in pairs:
        reviewer = solvers[reviewer_idx]["agent"]
        target = solvers[target_idx]["agent"]
        target_text = solvers[target_idx]["text"]
        
        print(f" {reviewer} is reviewing {target}...")
        critique = review_solution(problem, target_text)
        
        reviews.append({
            "reviewer": reviewer,
            "target": target,
            "critique_object": critique.dict()
        })
        
    return {"reviews_data": reviews}

def node_refinement(state: DebateState):
    """Stage 3: Solvers fix their answers."""
    print("\n--- STAGE 3: REFINEMENT ---")
    problem = state["problem_text"]
    solvers = state["solvers_data"]
    reviews = state["reviews_data"]
    refined_results = []
    
    review_map = {0: 2, 1: 0, 2: 1}
    
    for i in range(3):
        agent_name = solvers[i]["agent"]
        original_text = solvers[i]["text"]
        
        reviewer_idx = review_map[i]
        critique_data = reviews[reviewer_idx]["critique_object"]
        critique_text = str(critique_data["weaknesses"]) 
        
        print(f" {agent_name} is fixing their solution...")
        refined = refine_solution(problem, original_text, critique_text) # Returns RefinedSolution Object
        
        refined_results.append({
            "agent": agent_name,
            "final_text": refined.final_solution,
            "final_answer": refined.final_answer,
            "changes": refined.changes_made # Save this too!
        })
        
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