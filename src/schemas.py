from pydantic import BaseModel, Field
from typing import List, Optional

# --- STAGE 0: ROLE ASSIGNMENT ---
class RolePreference(BaseModel):
    role_preference: str = Field(description="Preferred Role: 'Solver' or 'Judge'")
    solver_confidence: float = Field(description="Confidence score (0-1) for being a Solver")
    judge_confidence: float = Field(description="Confidence score (0-1) for being a Judge")
    reasoning: str = Field(description="Why I am best suited for this role")

# --- STAGE 1: INITIAL SOLUTION ---
class InitialSolution(BaseModel):
    solution_text: str = Field(description="The full detailed solution") 
    step_by_step_reasoning: str = Field(description="Numbered list of steps taken")
    final_answer_short: str = Field(description="The concise final answer (e.g. '223')")
    
# --- STAGE 2: PEER REVIEW ---
class PeerReview(BaseModel):
    strengths: List[str] = Field(description="List of things the solver did right")
    weaknesses: List[str] = Field(description="List of logical gaps or errors")    
    error_severity: str = Field(description="Critical, Minor, or None")
    score: int = Field(description="Score out of 10")

# --- STAGE 3: REFINED SOLUTION ---
class RefinedSolution(BaseModel):
    changes_made: str = Field(description="Summary of what was fixed based on feedback")
    final_solution: str = Field(description="The updated, corrected solution")
    final_answer: str = Field(description="Definitive final asnwer")

# --- STAGE 4: FINAL JUDGMENT ---
class FinalJudgment(BaseModel):
    winner_id: str = Field(description="Which olver won? (e.g. 'Solver 1')")
    confidence: float = Field(description="0.0 to 1.0")
    reasoning: str = Field(description="Why this solution is better than the others")
    final_answer_rext: str = Field(description="The canonical answer to return to user")


