# Model Context Window Database
# Maps model names to their context window sizes and recommended reset intervals

MODEL_SPECS = {
    # DeepSeek Models (Large Context)
    "deepseek/deepseek-v3.1-terminus": {
        "context_window": 64000,
        "recommended_reset_interval": 25,
        "min_reset_interval": 15
    },
    "deepseek/deepseek-chat": {
        "context_window": 64000,
        "recommended_reset_interval": 25,
        "min_reset_interval": 15
    },
    
    # Grok Models
    "x-ai/grok-code-fast-1": {
        "context_window": 8192,
        "recommended_reset_interval": 8,
        "min_reset_interval": 5
    },
    
    # GPT Models
    "openai/gpt-4": {
        "context_window": 8192,
        "recommended_reset_interval": 8,
        "min_reset_interval": 5
    },
    "openai/gpt-4-turbo": {
        "context_window": 128000,
        "recommended_reset_interval": 30,
        "min_reset_interval": 20
    },
    "openai/gpt-3.5-turbo": {
        "context_window": 4096,
        "recommended_reset_interval": 5,
        "min_reset_interval": 3
    },
    
    # Claude Models
    "anthropic/claude-3-opus": {
        "context_window": 200000,
        "recommended_reset_interval": 40,
        "min_reset_interval": 25
    },
    "anthropic/claude-3-sonnet": {
        "context_window": 200000,
        "recommended_reset_interval": 40,
        "min_reset_interval": 25
    },
    "anthropic/claude-3-haiku": {
        "context_window": 200000,
        "recommended_reset_interval": 40,
        "min_reset_interval": 25
    },
    
    # Gemini Models
    "google/gemini-pro": {
        "context_window": 32000,
        "recommended_reset_interval": 15,
        "min_reset_interval": 10
    },
    
    # Local/Smaller Models
    "mistral/mistral-7b-instruct": {
        "context_window": 8192,
        "recommended_reset_interval": 8,
        "min_reset_interval": 5
    },
    "meta-llama/llama-2-13b-chat": {
        "context_window": 4096,
        "recommended_reset_interval": 5,
        "min_reset_interval": 3
    },
    
    # Default fallback for unknown models
    "default": {
        "context_window": 4096,
        "recommended_reset_interval": 5,
        "min_reset_interval": 3
    }
}

def get_model_specs(model_name: str):
    """
    Get context window specs for a given model.
    Returns default specs if model not found.
    """
    # Try exact match
    if model_name in MODEL_SPECS:
        return MODEL_SPECS[model_name]
    
    # Try partial match (e.g., "gpt-4" in "openai/gpt-4-0125")
    for key in MODEL_SPECS.keys():
        if key in model_name or model_name in key:
            return MODEL_SPECS[key]
    
    # Return default
    return MODEL_SPECS["default"]

def calculate_dynamic_reset_interval(model_name: str, conversation_history: list) -> int:
    """
    Dynamically calculate when to reset memory based on:
    1. Model's context window
    2. Current conversation length
    
    Returns recommended number of steps before reset.
    """
    specs = get_model_specs(model_name)
    
    # Estimate tokens per turn (rough: ~500 tokens per turn including system prompts)
    avg_tokens_per_turn = 500
    
    # Use 80% of context window as safety margin
    safe_context = specs["context_window"] * 0.8
    
    # Calculate how many turns fit
    max_turns = int(safe_context / avg_tokens_per_turn)
    
    # Clamp between min and recommended
    recommended = specs["recommended_reset_interval"]
    minimum = specs["min_reset_interval"]
    
    # If conversation is already large, use smaller interval
    if len(conversation_history) > max_turns * 0.7:
        return minimum
    
    return min(recommended, max_turns)
