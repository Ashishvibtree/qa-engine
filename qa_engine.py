import os
import json
import re
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from typing import List, Literal

# --- Bulletproof Structured Output Schema ---
class CapturedField(BaseModel):
    field_name: str = Field(description="Name of the field (e.g., 'budget', 'timeline', 'intent')")
    extracted_value: str = Field(description="The value captured, or 'Not Captured'")

class QASection(BaseModel):
    category: Literal[
        "Flow Compliance", 
        "Conversation Quality", 
        "Objection Handling", 
        "Lead Qualification", 
        "Call Outcome", 
        "Language & Tone",
        "Latency & Dead Air"
    ]
    status: Literal["Pass", "Fail", "Partial"]
    reasoning: str = Field(description="Strict analysis of why it received this status.")
    transcript_moments: List[str] = Field(description="Exact verbatim quotes from the transcript proving the point.")

class QAScorecard(BaseModel):
    overall_compliance_score: int = Field(ge=0, le=100)
    lead_classification: Literal["Interested", "Not Interested", "Callback", "Unqualified"]
    captured_fields: List[CapturedField] = Field(description="List of captured data points")
    sections: List[QASection]

# --- Engine ---
class QAEngine:
    def __init__(self):
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )
        self.model = "stepfun/step-3.5-flash"

    async def evaluate_call(self, transcript: str, agent_config: dict) -> str:
        
        schema_string = json.dumps(QAScorecard.model_json_schema(), indent=2)

        system_instructions = f"""
You are an elite, evidence-based Quality Assurance Evaluator for AI agents. Your task is to rigorously analyze a call transcript against the agent's system prompt and success criteria, then output a strict JSON scorecard.

### INPUT CONTEXT
<agent_prompt>
{agent_config.get('prompt', '')}
</agent_prompt>

<success_criteria>
{agent_config.get('success_criteria', '')}
</success_criteria>

### EVALUATION RUBRIC
Analyze the transcript across these exact 7 categories. For each, assign a status, provide concise reasoning, and cite EXACT verbatim quotes as evidence.

1. Flow Compliance: Verify sequence (Intro → Qualification → Pitch → Objection Handling → CTA/Close).
2. Conversation Quality: Assess contextual awareness vs. blind script-reading.
3. Objection Handling: Distinguish objections from questions; verify approved rebuttals.
4. Lead Qualification: Verify capture of required fields and assess signals.
5. Call Outcome: Evaluate closing and flag abrupt endings.
6. Language & Tone: Verify persona alignment.
7. Latency & Dead Air: Flag broken flow (If no timing data exists, mark Pass with "No timing data").

### SCORING & STATUS RULES
- Status: Pass, Partial, or Fail.
- overall_compliance_score (0-100): Deduct points proportionally for Partial/Fail sections.
- lead_classification: Must be EXACTLY one of: "Interested", "Not Interested", "Callback", "Unqualified".
- captured_fields: Extract key-value pairs. 

### CRITICAL JSON INSTRUCTIONS
You MUST output ONLY valid JSON. 
Do NOT wrap the JSON in markdown formatting (no ```json). Do not include any conversational text before or after the JSON.
Your JSON output MUST exactly match this JSON Schema structure:

{schema_string}
"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_instructions},
                    {"role": "user", "content": f"<transcript>\n{transcript}\n</transcript>"}
                ],
                # 🔥 FIX: Removed response_format entirely to bypass DeepInfra limitations
                temperature=0.0
            )
            
            raw_content = response.choices[0].message.content
            
            # 🔥 FIX: Python fallback to extract JSON if the model ignores the "no markdown" rule
            json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
            if json_match:
                return json_match.group(0)
            else:
                return raw_content
                
        except Exception as e:
            print(f"QA Evaluation Error: {e}")
            return None
