import inspect
from typing import Any, Dict, Callable

def handle_tool_execution_error(tool_name: str, exception: Exception, tool_func: Callable) -> Dict[str, Any]:
    """
    Analyzes a tool execution error and returns a helpful message for the agent.
    Uses introspection to provide signature hints.
    """
    error_msg = str(exception)
    
    # Get Tool Signature
    try:
        sig = inspect.signature(tool_func)
        sig_str = str(sig)
    except Exception:
        sig_str = "(parameters could not be determined)"

    # Base structured error
    response = {
        "status": "error",
        "error_type": type(exception).__name__,
        "original_message": error_msg,
        "tool_name": tool_name
    }

    # Guidance Logic
    guidance = ""
    
    if isinstance(exception, TypeError):
        if "missing" in error_msg and "argument" in error_msg:
            guidance = f"CRITICAL: You missed a required argument. Check the tool signature below."
        elif "unexpected keyword argument" in error_msg:
            guidance = f"CRITICAL: You passed an argument that the tool does not accept. Check the signature."
        else:
            guidance = "A type error occurred. Check if you are passing the correct data types."
    else:
        guidance = "An error occurred during tool execution."

    # Construct Final Helpful Message
    full_message = (
        f"Tool '{tool_name}' FAILED.\n"
        f"Error: {error_msg}\n"
        f"Guidance: {guidance}\n"
        f"EXPECTED SIGNATURE: {tool_name}{sig_str}\n"
        f"ACTION: Correct your tool call arguments and try again."
    )
    
    response["message"] = full_message
    return response
