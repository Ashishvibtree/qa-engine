import os
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from typing import List, Literal

# --- Structured Output Schema ---
class QASection(BaseModel):
    category: str = Field(description="e.g., 'Flow Compliance', 'Conversation Quality'")
    status: Literal["Pass", "Fail", "Partial"]
    reasoning: str = Field(description="Strict analysis of why it received this status.")
    transcript_moments: List[str] = Field(description="Quotes from the transcript proving the point.")

class QAScorecard(BaseModel):
    overall_compliance_score: int = Field(ge=0, le=100)
    lead_classification: Literal["Interested", "Not Interested", "Callback", "Unqualified"]
    captured_fields: dict = Field(description="Key-value pairs of captured data (budget, beds, intent, etc.)")
    sections: List[QASection]

# --- Engine ---
class QAEngine:
    def __init__(self):
        # OpenRouter setup (We use a smart model for reasoning)
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )
        self.model = "openai/gpt-4o-2024-08-06" # Best for strict JSON Pydantic parsing

    async def evaluate_call(self, transcript: str, agent_config: dict) -> str:
        
        system_instructions = f"""
        You are a brutal, highly accurate Quality Assurance Call Evaluator.
        Evaluate the AI Agent's performance strictly against its provided prompt and success criteria.

        [AGENT'S SYSTEM PROMPT]
        {agent_config['prompt']}

        [SUCCESS CRITERIA]
        {agent_config['success_criteria']}

        Evaluate on:
        1. Flow Compliance: Sequence followed? Mandatory questions asked?
        2. Conversation Quality: Contextual? Handled interruptions?
        3. Objection Handling: Right rebuttals used?
        4. Lead Qualification: Fields captured?
        5. Call Outcome: Did it end correctly?
        6. Language & Tone: Persona maintained?
        """

        try:
            response = await self.client.beta.chat.completions.parse(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_instructions},
                    {"role": "user", "content": f"[CALL TRANSCRIPT]\n{transcript}"}
                ],
                response_format=QAScorecard,
                temperature=0.1
            )
            
            return response.choices[0].message.content # Returns JSON string
        except Exception as e:
            print(f"QA Evaluation Error: {e}")
            return None