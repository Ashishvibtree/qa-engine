import os

def load_text_file(filename: str) -> str:
    """Reads a prompt from the prompts folder."""
    filepath = os.path.join("prompts", filename)
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        print(f"⚠️ Warning: Prompt file {filename} not found!")
        return ""

# Map agent names to their specific prompts and success criteria
AGENT_KNOWLEDGE_BASE = {
    "salma": {
        "prompt": load_text_file("salma_prompt.txt"),
        "success_criteria": """
            1. Must ask for a demo. 
            2. Must not list all features at once. 
            3. Must capture budget.
        """
    },
    "julia": {
        "prompt": load_text_file("julia_prompt.txt"),
        "success_criteria": """
            1. Must ask if they want a call or WhatsApp. 
            2. Only use first name. 
            3. Never say 'wait' or 'pause'.
        """
    },
    "dre": {
        "prompt": load_text_file("dre_prompt.txt"),
        "success_criteria": """
            1. Must find out if 4BR or 5BR. 
            2. Must quote correct price ranges. 
            3. Do not ask for WhatsApp until the end.
        """
    },
    "beacon": {
        "prompt": load_text_file("beacon_prompt.txt"),
        "success_criteria": """
            Define what a successful Beacon call looks like here.
        """
    }
}

def get_agent_config(retell_agent_name: str):
    """Fuzzy matching to find the right config based on the Retell Agent Name"""
    name_lower = retell_agent_name.lower()
    if "salma" in name_lower: return AGENT_KNOWLEDGE_BASE["salma"]
    if "julia" in name_lower: return AGENT_KNOWLEDGE_BASE["julia"]
    if "dre" in name_lower or "mia" in name_lower: return AGENT_KNOWLEDGE_BASE["dre"]
    if "beacon" in name_lower: return AGENT_KNOWLEDGE_BASE["beacon"]
    
    return None # Fallback if agent is unknown