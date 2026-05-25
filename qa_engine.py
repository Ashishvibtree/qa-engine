import os
import json
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
        # Using StepFun as requested, but this will now work with almost any model
        self.model = "stepfun/step-3.5-flash"

    async def evaluate_call(self, transcript: str, agent_config: dict) -> str:
        
        # 1. Dynamically extract our strict schema into a readable JSON string
        schema_string = json.dumps(QAScorecard.model_json_schema(), indent=2)

        # 2. Inject it directly into the System Prompt
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

1. Flow Compliance
   - Verify sequence: Intro → Qualification → Pitch → Objection Handling → CTA/Close.
   - Check if all mandatory questions/steps were asked.
   - Identify off-script deviations or missed steps.
2. Conversation Quality
   - Assess contextual awareness vs. blind script-reading.
   - Evaluate handling of interruptions, off-topic diversions, and recovery.
   - Flag unnecessary repetition or robotic phrasing.
3. Objection Handling
   - Distinguish objections from simple questions.
   - Verify use of approved rebuttals/prompt guidelines.
   - Assess persistence: Did it give up too early or push aggressively?
4. Lead Qualification
   - Verify capture of required fields (budget, timeline, intent, etc.).
   - Assess signal recognition (warm/cold cues).
5. Call Outcome
   - Evaluate closing: Proper CTA, next steps, or handoff?
   - Flag abrupt endings or failed transfers.
6. Language & Tone
   - Verify language consistency and persona alignment.
   - Assess professionalism, empathy, and pacing.
7. Latency & Dead Air
   - Flag long pauses, talking over the user, or broken flow.
   - If no timing data exists in the transcript, set status to "Pass" and reasoning to "No timing data available."

### SCORING & STATUS RULES
- Status Definitions:
  • Pass: Meets all core requirements for this category.
  • Partial: Meets some requirements but has notable gaps.
  • Fail: Misses critical requirements, breaks flow, or contradicts the prompt.
- overall_compliance_score (0-100): Calculate based on weighted performance. Flow Compliance & Lead Qualification carry the highest weight. Deduct points proportionally for Partial/Fail sections. Be strict but fair.
- lead_classification: Must be EXACTLY one of: "Interested", "Not Interested", "Callback", "Unqualified".
- captured_fields: Extract key-value pairs of qualified data points. 

### STRICT OUTPUT CONSTRAINTS
- transcript_moments MUST contain exact, verbatim quotes from the transcript. NEVER paraphrase or invent quotes. If no direct quote applies, use an empty list [].
- reasoning must be concise, analytical, and directly tied to the evidence.

### CRITICAL JSON INSTRUCTIONS
You MUST output ONLY valid JSON. Do NOT wrap the JSON in markdown code blocks (e.g., ```json).
Your JSON output MUST exactly match the following JSON Schema structure:

{schema_string}
"""

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_instructions},
                    {"role": "user", "content": f"<transcript>\n{transcript}\n</transcript>"}
                ],
                # Using standard JSON mode for universal model compatibility
                response_format={
                    "type": "json_object"
                },
                temperature=0.0
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"QA Evaluation Error: {e}")
            return None
