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

def get_agent_config(retell_agent_identifier: str):
    """Fuzzy matching to find the right config based on the Retell Agent Name OR Agent ID"""
    if not retell_agent_identifier:
        return None
        
    identifier_lower = retell_agent_identifier.lower()
    
    # Explicitly map the exact Agent ID for Salma from your Retell Campaign
    if "Salma-SLM" in identifier_lower or "8e9b2bc6-4667-49c2-8ef8-bd11d733e3f1" in identifier_lower: 
        return AGENT_KNOWLEDGE_BASE["salma"]
        
    if "julia" in identifier_lower: 
        return AGENT_KNOWLEDGE_BASE["julia"]
        
    if "dre" in identifier_lower or "mia" in identifier_lower: 
        return AGENT_KNOWLEDGE_BASE["dre"]
        
    if "beacon" in identifier_lower: 
        return AGENT_KNOWLEDGE_BASE["beacon"]
    
    # Safe fallback: If it's truly unknown, skip it rather than guessing wrongly.
    return None
