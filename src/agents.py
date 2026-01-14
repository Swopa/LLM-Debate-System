from langchain_core.prompts import ChatPromptTemplate
from src.config import get_llm
from src.schemas import (
    InitialSolution,
    PeerReview,
    RefinedSolution,
    FinalJudgment,
    RolePreference
)
from src.prompts import (
    SOLVER_PROMPT,
    REVIEWER_PROMPT,
    REFINER_PROMPT,
    JUDGE_PROMPT
)

# --- TEMPLATE PATTERN ---
#We definde templates once, using {variable_name} syntax.

# --- STAGE 0: ROLE ASSIGNMENT ---

def get_role_preference(problem_text: str, agent_name: str) -> RolePreference : 
    """
    Agent self-assesses: Am I better at Solving or Judging this specific problem?   
    """
    llm = get_llm(temperature=0.7)
    structured_llm = llm.with_structured_output(RolePreference)

    system_prompt = """You are an AI participant in a reasoning debate.
    
    OPTIONS:
    1. "Solver": Choose this if you are confident you can derive the answer step-by-step.
    2. "Judge": Choose this if you are better at evaluating others' logic than generating it yourself.
    
    Analyze the problem difficulty and your capabilities. Return your preference and confidence scores.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "PROBLEM: {problem}\n\nIdentity: {agent_name}")
    ])

    chain = prompt | structured_llm
    return chain.invoke({"problem": problem_text, "agent_name": agent_name})

# --- SOLVERS ---
def generate_solution(problem_text: str, agent_id: str) -> InitialSolution:
    llm = get_llm(temperature = 0.7)
    structured_llm = llm.with_structured_output(InitialSolution)

    # 1. Define Templates with placeholder
    prompt = ChatPromptTemplate.from_messages([
        ("system", SOLVER_PROMPT),
        ("human", "PROBLEM: {problem}\n\nYou are {agent_id}. Solve this.")
    ])
    
    chain = prompt | structured_llm

    # 2. Pass Values Safely
    return chain.invoke({"problem": problem_text, "agent_id": agent_id})

# --- THE REVIEWER PHASE ---
def review_solution(problem_text: str, solution_text: str) -> PeerReview:
    llm = get_llm(temperature=0.5)
    structured_llm = llm.with_structured_output(PeerReview)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", REVIEWER_PROMPT),
        ("human", "PROBLEM: {problem}\n\nPROPOSED SOLUTION:\n{solution}")
    ])
    
    chain = prompt | structured_llm
    
    # Pass values safely
    return chain.invoke({"problem": problem_text, "solution": solution_text})

# --- THE REFINEMENT ---
def refine_solution(problem_text: str, original_solution: str, critique: str) -> RefinedSolution:
    llm = get_llm(temperature=0.6)
    structured_llm = llm.with_structured_output(RefinedSolution)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", REFINER_PROMPT),
        ("human", """
        PROBLEM: {problem}
        
        YOUR ORIGINAL SOLUTION:
        {original}
        
        CRITIQUE RECEIVED:
        {critique}
        
        Fix your solution now.
        """)
    ])

    chain = prompt | structured_llm
    return chain.invoke({
        "problem": problem_text,
        "original": original_solution,
        "critique": critique
    })

# --- THE JUGE ---
def judge_debate(problem_text: str, solutions: list) -> FinalJudgment:
    llm = get_llm(temperature=0.2)
    structured_llm = llm.with_structured_output(FinalJudgment)

    # Pre-format solution
    solutions_str = ""
    for i, sol in enumerate(solutions):
        #We escape braces just in case
        safe_sol = sol.replace("{", "{{").replace("}", "}}")
        solutions_str += f"\n--- SOLUTION {i+1} ---\n{safe_sol}\n"

    prompt = ChatPromptTemplate.from_messages([
        ("system", JUDGE_PROMPT),
        ("human", "PROBLEM: {problem}\n\nCANDIDATE SOLUTIONS:\n{candidates}")
    ])
    
    chain = prompt | structured_llm
    return chain.invoke({
        "problem": problem_text, 
        "candidates": solutions_str
    })