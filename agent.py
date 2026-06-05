# agent.py
import json
import os
import re
import traceback
import concurrent.futures
import time
import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Callable

import openai
import mlflow
import pandas as pd

from state_manager import PERSISTENT_PATHS
from ui_logger import log
# Cognitive Refactor Imports
from cluster.cognitive_tasks import run_analyst_task, run_strategist_task
from cluster.mlflow_utils import MLFlowAgentContext, log_agent_artifact
from cluster.trace_logger import save_agent_trace




class ConversationalAgent:
    """
    Manages the interaction with an LLM via an OpenAI-compatible API.
    """

    # Models that use extended thinking — require temperature=1.0 and return
    # the response in reasoning_content rather than content.
    # Also matches local path-based model names (e.g. /mnt/.../Qwen3.5-35B-A3B).
    _THINKING_MODEL_PREFIXES = ("qwen/qwen3", "deepseek/deepseek-r1", "deepseek-r1")
    _THINKING_MODEL_KEYWORDS = ("qwen3", "deepseek-r1")

    def __init__(self, base_url: str, api_key: str, model_name: str):
        """
        Initializes the ConversationalAgent with specific API credentials.

        Args:
            base_url (str): The base URL of the API endpoint.
            api_key (str): The API key for authentication.
            model_name (str): The name of the model to use.
        """
        try:
            self.model_name = model_name
            self.is_thinking_model = any(
                model_name.lower().startswith(p) for p in self._THINKING_MODEL_PREFIXES
            ) or any(
                k in model_name.lower() for k in self._THINKING_MODEL_KEYWORDS
            )
            self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
            log("[AGENTE] Agent instance created.")
            log(f"[AGENTE] -> Provider URL: {base_url}")
            log(f"[AGENTE] -> Using model: {self.model_name}")
            if self.is_thinking_model:
                log(f"[AGENTE] -> Thinking model detected: temperature=1.0, reasoning_content fallback enabled.")

        except Exception as e:
            log(f"[AGENTE] CRITICAL: Failed to initialize OpenAI client: {e}")
            raise AgentCriticalError(f"Error configuring the OpenAI client: {e}")

    def run_turn(self, system_prompt: str, conversation_history: List[Dict[str, Any]], model_override: str = None) -> Tuple[str, Optional[str]]:
        """
        Executes a single turn of conversation with the LLM.
        Handles tool calls enclosed in <tool_code|tool_call|function_call> tags
        AND attempts to parse raw JSON output if no tags are found, even if mixed with text.
        """
        log("[AGENTE] Starting reasoning turn...")

        # --- Build Message List (No Changes Here) ---
        messages = [{"role": "system", "content": system_prompt}]
        for msg in conversation_history:
            role = msg.get("role", "user")
            if role == "model":
                role = "assistant"
            parts = msg.get("parts")
            if parts and isinstance(parts, list) and len(parts) > 0:
                content = parts[0]
            else:
                content = msg.get("content", "")
            if content:
                messages.append({"role": role, "content": content})
            else:
                log(f"[AGENTE] WARNING: Skipping malformed message in history: {msg}")

        # --- COMPATIBILITY FIX: Ensure at least one USER message exists ---
        # Some providers (e.g. Together via OpenRouter) reject system-only requests.
        if len(messages) == 1 and messages[0]["role"] == "system":
            messages.append({
                "role": "user", 
                "content": "Please analyze the provided context and instructions, then proceed with the requested output."
            })
            log("[AGENTE] Added default user message for provider compatibility.")

        # --- API Call and Error Handling with Retry ---
        max_retries = 5
        base_delay = 2
        
        for attempt in range(max_retries):
            try:
                # log(f"[AGENTE] Sending complete prompt context:\n{json.dumps(messages, indent=2)}") #comment it to remove the prompts
                log(f"[AGENTE] Sending messages to model (Attempt {attempt+1}/{max_retries})...")
                
                target_model = model_override if model_override else self.model_name
                if model_override:
                    log(f"[AGENTE] Overriding model to: {target_model}")

                # Thinking models need temperature=1.0; standard models use 0.7.
                is_thinking = self.is_thinking_model or (
                    model_override and any(
                        model_override.lower().startswith(p)
                        for p in self._THINKING_MODEL_PREFIXES
                    )
                )
                api_kwargs = dict(
                    model=target_model,
                    messages=messages,
                    temperature=1.0 if is_thinking else 0.7,
                )
                if is_thinking:
                    # Ask OpenRouter to enable thinking with a bounded token budget.
                    api_kwargs["extra_body"] = {
                        "thinking": {"type": "enabled", "budget_tokens": 4096}
                    }
                response = self.client.chat.completions.create(**api_kwargs)

                # Thinking models may return content=None with text in reasoning_content.
                if not response or not response.choices:
                    raise ValueError(f"Empty or null response from model (choices={getattr(response, 'choices', None)})")
                msg = response.choices[0].message
                raw_response = (
                    msg.content
                    or getattr(msg, "reasoning_content", None)
                    or getattr(msg, "reasoning", None)
                    or ""
                )
                # Strip reasoning blocks emitted inline by local reasoning models (e.g. Qwen3.5 via vllm).
                # Handles both <think>...</think> and orphan </think> (opening tag omitted by vllm).
                if "</think>" in raw_response:
                    raw_response = raw_response.split("</think>", 1)[-1].strip()
                else:
                    raw_response = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL).strip()
                log(f"[AGENTE] Raw response generated:\n{raw_response}")
                break # Success!
            
            except (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError) as e:
                log(f"[AGENTE] WARNING: API Connection/Timeout Error (Attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    sleep_time = base_delay * (2 ** attempt) + (random.random() * 0.5) # Jitter
                    log(f"[AGENTE] Retrying in {sleep_time:.2f}s...")
                    time.sleep(sleep_time)
                else:
                     log(f"[AGENTE] ERROR: Max retries reached. API Call Failed.")
                     return f"Failed to connect to the LLM server after {max_retries} attempts. Error: {e}", None
            
            except Exception as e:
                log(f"[AGENTE] ERROR: An unexpected error occurred during model call: {e}")
                return f"An unexpected error occurred: {e}", None

        # --- START UPDATED EXTRACTION LOGIC ---
        tool_call_content = None

        # 1. Try to find content within standard tags first
        tag_match = re.search(
            r"<(tool_code|tool_call|function_call)>(.*?)</\1>", raw_response, re.DOTALL | re.IGNORECASE
        )

        if tag_match:
            # If tags are found, extract content from between them
            tag_name = tag_match.group(1)
            tool_call_content = tag_match.group(2).strip()
            log(f"[AGENTE] Extracted tool call content using tag: <{tag_name}>")
        else:
            # 2. Fallback: Look for a JSON array or object at the end of the string
            log("[AGENTE] No tool call tags found. Scanning for JSON structure...")
            
            # First, remove content inside thought tags to avoid false matches
            cleaned_response = re.sub(r"<thought_step_\d[^>]*>[\s\S]*?</thought_step_\d[^>]*>", "", raw_response)
            
            # Strategy: Prioritize arrays [...] over objects {...}
            # Arrays are more common for multi-tool calls
            
            # Try to find JSON array first
            array_match = re.search(r"\[[\s\S]*\]", cleaned_response)
            if array_match:
                potential_json = array_match.group(0)
                try:
                    parsed = json.loads(potential_json)
                    # Validate it's actually a tool call structure
                    if isinstance(parsed, list) and len(parsed) > 0:
                        if all(isinstance(item, dict) and "tool_name" in item for item in parsed):
                            log(f"[AGENTE] Found valid JSON array with {len(parsed)} tool calls.")
                            tool_call_content = potential_json
                        else:
                            log("[AGENTE] Found JSON array but structure doesn't match tool calls. Ignoring.")
                    else:
                        log("[AGENTE] Found empty or invalid JSON array. Trying objects...")
                except json.JSONDecodeError:
                    log("[AGENTE] Found array-like content but JSON parsing failed.")
            
            # If no valid array found, try single object
            if not tool_call_content:
                obj_match = re.search(r"\{[\s\S]*\}", cleaned_response)
                if obj_match:
                    potential_json = obj_match.group(0)
                    try:
                        parsed = json.loads(potential_json)
                        # Validate it's a tool call object
                        if isinstance(parsed, dict) and "tool_name" in parsed:
                            log("[AGENTE] Found valid single tool call object.")
                            tool_call_content = potential_json
                        else:
                            log("[AGENTE] Found JSON object but doesn't look like a tool call.")
                    except json.JSONDecodeError:
                        log("[AGENTE] Found object-like content but JSON parsing failed.")
            
            if not tool_call_content:
                log("[AGENTE] No valid tool call JSON found in response.")

        # Return the FULL raw response and the potentially extracted tool call content
        return raw_response, tool_call_content
        # --- END UPDATED EXTRACTION LOGIC ---

    def summarize_long_term_memory(self, raw_memory: str) -> str:
        """
        Uses the LLM to summarize retrieved memory, reducing token count.
        Synchronous version for use in tools.
        """
        from cluster.prompts import MEMORY_SUMMARIZER_PROMPT
        try:
            summary, _ = self.run_turn(
                MEMORY_SUMMARIZER_PROMPT,
                [{"role": "user", "parts": [f"Raw Memory Retrieval:\n{raw_memory}"]}]
            )
            return summary
        except Exception as e:
            log(f"[MemorySummarizer] Failed: {e}")
            return raw_memory[:1000] + "... (Truncated due to error)"

    # In agent.py
    def parse_tool_calls(self, tool_call_str: str) -> List[Dict[str, Any]]:
        """
        Parses a JSON string that can contain a single tool call or a list of tool calls.
        Supports multiple formats and returns a list of tool call dictionaries.
        """
        log(f"[PARSER] Attempting to parse string for multiple tool calls: {tool_call_str}")

        json_content = None
        # --- 1. Extract content from known tags if present ---
        match = re.search(r"<(tool_code|tool_call|function_call)>(.*?)</\1>", tool_call_str, re.DOTALL | re.IGNORECASE)
        if match:
            tag = match.group(1)
            log(f"[PARSER] Found tag format: <{tag}>")
            json_content = match.group(2).strip()
        else:
            json_content = tool_call_str.strip()

        # --- 2. Remove Markdown-style fences and code labels ---
        json_content = re.sub(r"```(?:json|tool_code|tool_call)?\s*", "", json_content, flags=re.IGNORECASE)
        json_content = re.sub(r"```", "", json_content)

        # --- 3. Extract first JSON-looking block (either object or array) ---
        if ("{" in json_content and "}" in json_content) or ("[" in json_content and "]" in json_content):
            # This regex is a bit more general to find either a {...} or a [...] block
            # AUDIT FIX: Changed to avoid FutureWarning and potential nesting issues
            possible_json = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", json_content)
            if possible_json:
                json_content = possible_json.group(0)

        # --- 4. Clean trailing commas ---
        cleaned_str = re.sub(r",\s*(?=[}\]])", "", json_content)

        # --- 4b. Normalize Python literals to valid JSON ---
        # Some LLMs (e.g. Mistral) emit Python booleans/None instead of JSON true/false/null
        cleaned_str = re.sub(r'\bTrue\b', 'true', cleaned_str)
        cleaned_str = re.sub(r'\bFalse\b', 'false', cleaned_str)
        cleaned_str = re.sub(r'\bNone\b', 'null', cleaned_str)

        tool_data_list = []
        try:
            # --- 5. PRIMARY ATTEMPT: Parse as JSON ---
            parsed_data = json.loads(cleaned_str)
            if isinstance(parsed_data, list):
                tool_data_list = parsed_data
                log(f"[PARSER] Successfully parsed as a list of {len(tool_data_list)} tool calls.")
            elif isinstance(parsed_data, dict):
                tool_data_list = [parsed_data]
                log("[PARSER] Successfully parsed as a single tool call object.")
            else:
                raise ValueError("Parsed JSON is not a list or a dictionary.")

        except (json.JSONDecodeError, ValueError) as e:
            log(f"[PARSER] JSON parsing failed (Error: {e}). Checking for XML-style fallback for a single tool.")
            tool_name_match = re.search(r"<tool_name>(.*?)</tool_name>", cleaned_str, re.DOTALL)
            args_match = re.search(r"<(?:args|arguments)>(.*?)</(?:args|arguments)>", cleaned_str, re.DOTALL)

            if tool_name_match and args_match:
                log("[PARSER] Found XML-style tags. Manually constructing single tool_data.")
                try:
                    tool_name = tool_name_match.group(1).strip()
                    args_str = args_match.group(1).strip()
                    args_dict = json.loads(args_str)
                    tool_data_list = [{"tool_name": tool_name, "args": args_dict}]
                except json.JSONDecodeError as e2:
                    log(
                        f"[PARSER] ERROR: Found XML tags, but <args> content was not valid JSON: {e2}. Original args: '{args_str}'"
                    )
                    return []
            else:
                log(
                    f"[PARSER] ERROR: Fallback failed. Not valid JSON or XML-style. Original string: '{tool_call_str}'"
                )
                return []

        # --- 6. Process each tool call in the list ---
        final_tool_calls = []
        for tool_data in tool_data_list:
            try:
                tool_name = tool_data.get("tool_name") or tool_data.get("name")
                if not tool_name:
                    raise ValueError("Missing 'tool_name' or 'name' field in a tool_data object.")
                tool_name = tool_name.strip().replace("()", "")

                final_args = tool_data.get("args", {})
                if not isinstance(final_args, dict):
                    log(
                        f"[PARSER] WARNING: 'args' in tool call '{tool_name}' is not a dictionary. Defaulting to empty."
                    )
                    final_args = {}

                final_tool_calls.append({"tool_name": tool_name, "args": final_args})
                log(f"[PARSER] Processed tool: '{tool_name}', Args: {final_args}")

            except (ValueError, TypeError, AttributeError) as e:
                log(f"[PARSER] ERROR: Failed to process a tool_data object. Error: {e}. Data: '{tool_data}'")
                continue  # Skip this invalid tool call and proceed with others

        log(f"[PARSER] Parse complete. Found {len(final_tool_calls)} valid tool calls.")
        return final_tool_calls


import asyncio
import threading
# from multiprocessing import Manager # Removed

from system_utils import get_gpu_memory_info


class DeciderOrchestrator:
    def __init__(self, main_agent: ConversationalAgent, execute_tool_func: callable, num_gpus: int = 1, gpu_ids: List[int] = None):
        self.main_agent = main_agent
        self.execute_tool_func = execute_tool_func
        
        # --- GPU Resource Allocator ---
        if gpu_ids:
            self.gpu_pool = list(gpu_ids)
            self.num_gpus = len(gpu_ids)
            log(f"[ResourceAllocator] Initializing for specific GPUs: {gpu_ids}")
        else:
            self.num_gpus = num_gpus
            self.gpu_pool = list(range(num_gpus)) 
            log(f"[ResourceAllocator] Initializing for {num_gpus} GPUs (Range 0-{num_gpus-1}).")

        self.gpu_semaphore = threading.Semaphore(self.num_gpus)
        self.gpu_lock = threading.Lock()

    def _inject_new_architecture_prompt(self, thread_safe_state: Dict[str, Any]) -> str:
        """
        Checks if a new architecture has been created and injects a high-priority prompt.
        """
        if not thread_safe_state:
            return ""
        
        new_arch_file = thread_safe_state.get("latest_created_architecture")
        if new_arch_file:
            # Clear it so we don't nag forever, or keep it until used? 
            # Better to show it once clearly.
            # actually, maybe we shouldn't clear it immediately if the agent ignores it?
            # But for now let's show it.
            
            # We can peek at 'manifest_data' to suggest available representation if possible, 
            # but for now let's be generic and forceful.
            from cluster.prompts import NEW_ARCHITECTURE_MESSAGE
            msg = NEW_ARCHITECTURE_MESSAGE.format(new_arch_file=new_arch_file)
            # Optional: Clear the flag so next turn doesn't repeat it?
            # thread_safe_state.pop("latest_created_architecture", None) 
            # Let's NOT clear it here validation might fail. 
            # Ideally the tool 'run_training_trial' should clear it upon success.
            # But simpler logic: just show it if present. Thread Safe State is persistent?
            # If we don't clear it, it will show ever turn.
            # Let's clear it to avoid repetition loop.
            thread_safe_state.pop("latest_created_architecture", None)
            return msg
            
        return ""

    async def _run_gpu_tool_wrapper(self, tool_call: Dict[str, Any], thread_safe_state: Dict[str, Any]):
        """
        A wrapper that acquires a GPU, executes the tool, and releases the GPU.
        """
        gpu_id = -1
        log(f"[ResourceAllocator] GPU tool '{tool_call.get('tool_name')}' is waiting for a free GPU...")
        await asyncio.to_thread(self.gpu_semaphore.acquire)
        try:
            with self.gpu_lock:
                if self.gpu_pool:
                    # --- SMART GPU ALLOCATION (User Request) ---
                    try:
                        # 1. Get real-time stats
                        gpu_stats = get_gpu_memory_info()
                        # gpu_stats is like [{'total': 10000, 'used': 100, 'free': 9900}, ...]
                        
                        # 2. Filter available pool candidates
                        candidates = []
                        for gid in self.gpu_pool:
                            if 0 <= gid < len(gpu_stats):
                                candidates.append((gid, gpu_stats[gid]["free"]))
                            else:
                                # Fallback if index out of range (unlikely if configured correctly)
                                candidates.append((gid, -1))
                        
                        # 3. Sort by Free Memory (Descending) -> Best is first
                        candidates.sort(key=lambda x: x[1], reverse=True)
                        
                        if candidates:
                            best_gpu_id = candidates[0][0]
                            free_mem = candidates[0][1]
                            
                            # Remove specific ID from pool
                            self.gpu_pool.remove(best_gpu_id)
                            gpu_id = best_gpu_id
                            
                            log(f"[ResourceAllocator] Smart Select: Picked GPU {gpu_id} ({free_mem} MiB Free) from {len(candidates)} candidates.")
                        else:
                             # Should not happen if self.gpu_pool was not empty
                             gpu_id = self.gpu_pool.pop(0)

                    except Exception as e:
                        log(f"[ResourceAllocator] Smart Allocation Failed ({e}). Falling back to FIFO.")
                        if self.gpu_pool:
                            gpu_id = self.gpu_pool.pop(0)
                    # -------------------------------------------

            if gpu_id == -1:
                 log("[ResourceAllocator] CRITICAL: Semaphore acquired but pool empty!")
                 return {"status": "error", "message": "Resource allocation failed."}

            log(f"[ResourceAllocator] GPU {gpu_id} acquired for tool '{tool_call.get('tool_name')}'")
            # Pass the gpu_id to the execution function
            result = await asyncio.to_thread(
                self.execute_tool_func,
                tool_call.get("tool_name"),
                tool_call.get("args", {}),
                thread_safe_state,
                gpu_id=gpu_id  # Pass the allocated GPU ID
            )
            return result
        except Exception as e:
            log(f"[ResourceAllocator] CRITICAL ERROR during wrapped GPU tool execution: {e}")
            return {"status": "error", "message": f"Error during execution on GPU {gpu_id}: {e}"}
        finally:
            if gpu_id != -1:
                with self.gpu_lock:
                    self.gpu_pool.append(gpu_id)
                log(f"[ResourceAllocator] GPU {gpu_id} released.")
            self.gpu_semaphore.release()

    async def run_async_analysis(self, initial_context: str, thread_safe_state: Dict[str, Any]) -> str:
        # --- AUTOMATED DATA GATHERING (SMART FILTERING) ---
        log("[Analyst] Automating data gathering (Pre-computation) with Smart Filtering...")
        
        # Recuperiamo il modello corrente dallo stato sicuro
        # Recuperiamo il modello corrente dallo stato sicuro
        current_model = thread_safe_state.get("selected_model")
        
        # Custom Architecture Override REMOVED to prevent Split-Brain state.
        # The Analyst will now strictly follow 'thread_safe_state.get("selected_model")'.
        
        # Definiamo i tool di analisi standard da eseguire
        analysis_tools_requests = [
            {"tool_name": "get_recent_trials", "args": {"n": 5}},
            {"tool_name": "analyze_best_vs_worst_trials", "args": {}},
            {"tool_name": "get_optimization_status", "args": {}},
            {"tool_name": "get_correlation_matrix", "args": {}} 
        ]
        
        gathered_data = []
        
        for tool_req in analysis_tools_requests:
            tool_name = tool_req["tool_name"]
            args = tool_req["args"].copy() # Copia per non modificare l'originale per i cicli successivi
            
            # --- SMART FILTER INJECTION ---
            # Se il tool supporta 'architecture_filter' e non è stato specificato,
            # lo forziamo al modello corrente per evitare Context Overflow (scaricare troppi dati).
            if tool_name in ["get_recent_trials", "analyze_best_vs_worst_trials", "get_correlation_matrix"]:
                if "architecture_filter" not in args or args["architecture_filter"] is None:
                    if current_model:
                        args["architecture_filter"] = current_model
                        log(f"[Analyst] Auto-injected filter for {tool_name}: {current_model}")
            # ------------------------------

            try:
                # Eseguiamo il tool in modo sincrono (sono tool di analisi veloce su CPU)
                # Nota: self.execute_tool_func è il wrapper 'recursive_execute_tool' passato dal main
                result = self.execute_tool_func(tool_name, args, thread_safe_state)
                
                # Formattiamo il risultato per il prompt, indicando quale filtro è stato usato
                filter_used = args.get('architecture_filter', 'Global')
                gathered_data.append(f"--- {tool_name} Results ({filter_used}) ---\n{json.dumps(result, indent=2)}")
                
            except Exception as e:
                log(f"[Analyst] Auto-tool {tool_name} failed: {e}")

        # Costruiamo il contesto completo per l'LLM
        full_context = f"{initial_context}\n\n" + "\n\n".join(gathered_data)

        # --- 2. LLM Summary ---
        from cluster.prompts import ANALYST_SYSTEM_PROMPT
        
        log("[Multi-Agent-Async] Calling the Analyst for factual summary...")
        log("[Analyst] Starting LLM call...")
        
        try:
            # Chiamata all'LLM con il contesto filtrato e pulito
            # Usiamo asyncio.to_thread perché self.main_agent.run_turn è sincrono (bloccante)
            analysis, _ = await asyncio.to_thread(
                self.main_agent.run_turn,
                ANALYST_SYSTEM_PROMPT,
                [{"role": "user", "parts": [f"Here is the data context for {current_model}:\n{full_context}"]}]
            )
            log("[Analyst] LLM call finished.")
        except Exception as e:
            log(f"[Analyst] CRITICAL ERROR: {e}")
            analysis = "Analyst failed to generate summary due to an error (possibly context overflow)."
            
        return analysis

    async def summarize_long_term_memory(self, raw_memory: str) -> str:
        """
        Uses the LLM to summarize retrieving memory, reducing token count.
        """
        # LLM call to summarize
        from cluster.prompts import MEMORY_SUMMARIZER_PROMPT
        try:
            # Correctly calling run_turn on self.main_agent
            summary, _ = await asyncio.to_thread(
                self.main_agent.run_turn,
                MEMORY_SUMMARIZER_PROMPT,
                [{"role": "user", "parts": [f"Raw Memory Retrieval:\n{raw_memory}"]}]
            )
            return summary
        except Exception as e:
            log(f"[MemorySummarizer] Failed: {e}")
            return raw_memory[:1000] + "... (Truncated due to error)"

    async def run_async_strategy(self, conversation_history: list, analyst_report: str, graph_advice: str, current_todos: list = None) -> str:
        # --- Filter History Specifically for the STRATEGIST ---
        log("[Multi-Agent-Async] Filtering conversation history for the Strategist...")
        filtered_history_for_strategist = []
        key_feedback_tools = [
            "run_training_trial",
            "run_strategic_optuna_sweep",
            "run_supernet_search",  # AUDIT FIX: Allow Strategist to see supernet results to prevent OOM error loops
            "construct_model_from_genotype", # NEW TOOL: Essential for Strategist visibility
            "get_optimization_status",
            "run_web_research",
            "generate_creative_hypotheses",
            "run_architecture_comparison",
            "propose_intelligent_architecture",
            "validate_architecture_file",
            "write_architecture_file",
        ]

        # Pre-calculate the index of the LAST Analyst summary to avoid duplicates/token explosion
        last_analyst_summary_idx = -1
        for idx, msg in enumerate(conversation_history):
            content = msg.get("parts", [""])[0] if "parts" in msg else msg.get("content", "")
            if content and content.strip().startswith("Here is the Analyst's summary:"):
                last_analyst_summary_idx = idx

        for i, msg in enumerate(conversation_history):
            role = msg.get("role")
            content = msg.get("parts", [""])[0] if "parts" in msg else msg.get("content", "")

            if role == "user":
                if content.strip().startswith("Here is the Analyst's summary:"):
                    # TOKEN OPTIMIZATION: Only keep the latest summary
                    if i == last_analyst_summary_idx:
                        filtered_history_for_strategist.append({"role": "user", "content": content})
                    # Else: Skip old summaries
                    continue
                if content.strip().startswith("<tool_result>"):
                    try:
                        result_str = content.replace("<tool_result>", "").replace("</tool_result>", "").strip()
                        result_data = json.loads(result_str)
                        tool_name_from_result = None
                        if i > 0:
                            prev_msg = conversation_history[i - 1]
                            prev_content = (
                                prev_msg.get("parts", [""])[0] if "parts" in prev_msg else prev_msg.get("content", "")
                            )
                            tool_call_match = re.search(
                                r"<(tool_code|tool_call|function_call)>(.*?)</\1>",
                                prev_content,
                                re.DOTALL | re.IGNORECASE,
                            )
                            if tool_call_match:
                                try:
                                    call_json_str = tool_call_match.group(2).strip()
                                    cleaned_call_str = re.sub(
                                        r"^```(json)?\s*|\s*```$", "", call_json_str, flags=re.DOTALL
                                    )
                                    cleaned_call_str = re.sub(r",\s*(?=[}\]])", "", cleaned_call_str)
                                    call_data = json.loads(cleaned_call_str)
                                    tool_name_from_result = call_data.get("tool_name") or call_data.get("name")
                                except (json.JSONDecodeError, IndexError):
                                    log(
                                        f"[Filter Warning] Could not parse preceding message to find tool name for result: {result_str}"
                                    )
                        is_key_tool = tool_name_from_result in key_feedback_tools
                        is_error = result_data.get("status") == "error"
                        if is_key_tool or is_error:
                            context_prefix = (
                                f"[Result from {tool_name_from_result}]:\n"
                                if tool_name_from_result
                                else "[Tool Result]:\n"
                            )
                            filtered_history_for_strategist.append(
                                {"role": "user", "content": context_prefix + content}
                            )
                    except (json.JSONDecodeError, AttributeError):
                        # --- AUDIT FIX: Relaxed Filtering for Supernet Results ---
                        # If the content contains "genotype", it's likely a structured result that failed strict JSON parsing
                        # but contains critical info. FORCE INCLUDE IT.
                        if '"status": "error"' in content or '"genotype":' in content or '"final_model_file":' in content: 
                            filtered_history_for_strategist.append(
                                {
                                    "role": "user",
                                    "content": "[Tool Result - Raw Content]:\n" + content,
                                }
                            )
                        else:
                            log(
                                f"[Filter Debug] Skipping unparsable/non-error tool result for Strategist: {content[:100]}..."
                            )
            elif role == "assistant" or role == "model":
                is_tool_call_tag = bool(
                    re.search(r"<(tool_code|tool_call|function_call)>", content, re.DOTALL | re.IGNORECASE)
                )
                if not is_tool_call_tag:
                    # --- FIX: SANITIZE HISTORY TO PREVENT PATTERN MATCHING LOOPS ---
                    # The main Decider output uses <thought_step_...> XML tags.
                    # If the Strategist sees these in history, it mimics them, causing "impersonation" loops.
                    # We must strip them to show only the "clean" content or reduce the pattern strength.
                    
                    # Regex to remove <thought_step...>...</thought_step...> blocks if we want to hide them entirely,
                    # OR just remove the tags to simple text. 
                    # Strategy: Flatten the history to simple text to break the XML schema pattern.
                    clean_content = content
                    clean_content = re.sub(r"</?thought_step_\d+.*?>", "", clean_content, flags=re.DOTALL | re.IGNORECASE)
                    
                    # Also remove the specific role headers if they exist in the text to avoid role confusion
                    clean_content = re.sub(r"\((Analyst|Strategist|Decider)[^)]*\)", "", clean_content)

                    filtered_history_for_strategist.append({"role": "assistant", "content": clean_content.strip()})

        log(
            f"[Multi-Agent-Async] Strategist history filtered. Original: {len(conversation_history)}, Filtered: {len(filtered_history_for_strategist)}"
        )

        # Summarize memory if it's too long
        if graph_advice and len(graph_advice) > 2000:
            log("[Strategist] Memory advice is long. Summarizing...")
            graph_advice = await self.summarize_long_term_memory(graph_advice)

        # Format the to-do list for the prompt
        todos_str = "No active to-do list."
        if current_todos:
            todos_str = ""
            for t in current_todos:
                status_mark = "x" if t.get("status") == "completed" else " "
                todos_str += f"- [{status_mark}] {t.get('task')}\n"

        from cluster.prompts import STRATEGIST_SYSTEM_PROMPT
        strategist_system_prompt = STRATEGIST_SYSTEM_PROMPT.format(
            analyst_report=str(analyst_report) if analyst_report else "No fresh intelligence provided.",
            graph_advice=str(graph_advice) if graph_advice else "No long-term memory advice available.",
            current_todos=todos_str.strip()
        )
        
        log("[Multi-Agent-Async] Calling the Strategist for a high-level goal...")
        log("[Strategist] Starting LLM call...")
        try:
            hypothesis, _ = await asyncio.to_thread(
                self.main_agent.run_turn,
                strategist_system_prompt,
                filtered_history_for_strategist
            )
            log("[Strategist] LLM call finished.")
        except Exception as e:
            log(f"[Strategist] CRITICAL ERROR: {e}")
            hypothesis = "Strategist failed to generate hypothesis."
            
        return hypothesis

    def _check_for_user_message(self, thread_safe_state: Dict[str, Any]) -> str:
        """
        Checks for a user message file (user_message.txt) and returns its content formatted for the prompt.
        Only runs if manual_chat_mode is enabled in config/state.
        """
        # check config via state proxy
        # Since main.py explicitly sets st.session_state.manual_chat_mode, 
        # we can just fetch it directly from the thread_safe_state proxy.
        manual_mode = thread_safe_state.get("manual_chat_mode", False)

        if not manual_mode:
            return ""

        # Anchor to the project root (same dir as agent.py) to be CWD-independent.
        # This ensures the path matches what send_message.py writes, regardless of
        # how the agent process was launched (e.g. `python cluster/main.py` from parent).
        _project_root = os.path.dirname(os.path.abspath(__file__))
        message_file = os.path.join(_project_root, "user_message.txt")
        if os.path.exists(message_file):
            try:
                with open(message_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                
                # Delete after reading to consume it
                os.remove(message_file)
                
                if content:
                    log(f"[Orchestrator] 📩 USER MESSAGE RECEIVED: {content}")
                    from cluster.prompts import USER_URGENT_MESSAGE
                    return USER_URGENT_MESSAGE.format(user_msg=content)
            except Exception as e:
                log(f"[Orchestrator] Error reading user message: {e}")
        
        return ""

    async def run_decider_centric_flow(
        self, system_prompt: str, conversation_history: list, initial_context: str, selected_model: str, metadata_info: str = None, fast_mode: bool = False, thread_safe_state: Dict[str, Any] = None, graph_planner_advice: str = None
    ) -> tuple[str, str | None]:
        """
        Orchestrates the Decider-centric workflow using Celery for parallel cognitive processing.
        All reasoning steps are logged to MLFlow.
        """
        
        # ANSI Color Codes for differentiation in logs
        BLUE = "\033[94m"
        MAGENTA = "\033[95m"
        GREEN = "\033[92m"
        RESET = "\033[0m"

        log(f"{GREEN}[Orchestrator] Starting Distributed Cognitive Cycle (Celery + MLFlow)...{RESET}")

        # --- 1. SETUP MLFLOW CONTEXT ---
        run_name = f"Turn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with MLFlowAgentContext(experiment_name="Agent_Brain", run_name=run_name) as active_run:
            if not active_run:
                log("[Orchestrator] WARNING: MLFlow run failed to start. Proceeding without MLFlow tracking.")
                run_id = "offline_run_" + datetime.now().strftime('%Y%m%d_%H%M%S')
            else:
                run_id = active_run.info.run_id
                log(f"[Orchestrator] MLFlow Run Active: {run_id}")
            
            # --- 2. DISPATCH ASYNC AGENTS ---
            # --- FIX: Create Serializable State Snapshot ---
            # SharedStateManager and DictConfig are not JSON serializable. 
            # We create a clean dictionary snapshot for the worker.
            state_snapshot = {
                "selected_model": thread_safe_state.get("selected_model"),
                "results_log": thread_safe_state.get("results_log", []),
                "global_step_counter": thread_safe_state.get("global_step_counter", 0),
                "architecture_flags": thread_safe_state.get("architecture_flags", {}),
                "best_accuracy": thread_safe_state.get("best_accuracy", 0.0),
                "optimization_target": thread_safe_state.get("optimization_target"),
                "latest_created_architecture": thread_safe_state.get("latest_created_architecture"),
                "todos": thread_safe_state.get("todos", [])
            }
            
            # Handle Config (OmegaConf -> Dict)
            cfg = thread_safe_state.get("cfg")
            try:
                from omegaconf import OmegaConf, DictConfig
                if isinstance(cfg, (DictConfig, list)):
                    state_snapshot["cfg"] = OmegaConf.to_container(cfg, resolve=True)
                elif isinstance(cfg, dict):
                    state_snapshot["cfg"] = cfg
            except ImportError:
                 log(f"[Orchestrator] OmegaConf not found. Passing cfg as-is (might fail serialization if not dict).")
                 state_snapshot["cfg"] = cfg
            except Exception as e:
                 log(f"[Orchestrator] Warning: Failed to convert cfg to dict: {e}")
                 # Proceed without cfg or with raw cfg
                 state_snapshot["cfg"] = None

            # --- 2. SEQUENTIAL DISPATCH (Analyst -> Strategist) ---
            # Capture the current tracking URI to ensure workers use the exact same one
            tracking_uri = mlflow.get_tracking_uri()
            log(f"[Orchestrator] Passing Tracking URI to workers: {tracking_uri}")

            # Step 1: Run Analyst (Facts)
            log(f"[Orchestrator] Step 1/2: Dispatching Analyst task...")
            analyst_async = run_analyst_task.delay(
                initial_context, 
                state_snapshot, 
                run_id, 
                model_name=self.main_agent.model_name,
                tracking_uri=tracking_uri
            )
            
            try:
                # Wait for Analyst Result
                analysis = analyst_async.get(timeout=120)
                log(f"[Orchestrator] Analyst Report Received:\n{analysis}")
            except Exception as e:
                log(f"[Orchestrator] Analyst Failed: {e}")
                analysis = f"Error retrieving analysis: {str(e)}"

            # Step 2: Run Strategist (Goal based on Facts)
            log(f"[Orchestrator] Step 2/2: Dispatching Strategist task with fresh intelligence and active plan...")
            # We pass the analyst_report to the Strategist so it can see the latest facts
            strategist_async = run_strategist_task.delay(
                conversation_history, 
                run_id, 
                model_name=self.main_agent.model_name,
                analyst_report=analysis, # <--- JOINT INTELLIGENCE INJECTION
                graph_advice=graph_planner_advice, # <--- NEW: Graph Memory Awareness
                current_todos=state_snapshot.get("todos", []), # <--- NEW: Active Plan Awareness
                tracking_uri=tracking_uri
            )
            
            try:
                # Wait for Strategist Result
                hypothesis = strategist_async.get(timeout=120)
                log(f"[Orchestrator] Strategist Plan Received:\n{hypothesis}")
            except Exception as e:
                log(f"[Orchestrator] Strategist Failed: {e}")
                hypothesis = f"Error retrieving strategy: {str(e)}"


            # --- 4. THE DECIDER (Final Decision) ---
            from cluster.prompts import DECIDER_SYSTEM_PROMPT
            decider_context = DECIDER_SYSTEM_PROMPT.format(
                selected_model=selected_model,
                metadata_info=metadata_info,
                num_gpus=self.num_gpus,
                analysis=analysis,
                hypothesis=hypothesis,
                graph_planner_advice=graph_planner_advice if graph_planner_advice else "**3. GRAPH PLANNER:** (Inactive or no suggestions)",
                new_architecture_prompt=self._inject_new_architecture_prompt(thread_safe_state),
                user_message_prompt=self._check_for_user_message(thread_safe_state)
            )
            log("[Orchestrator] Calling the Decider (Main LLM) for final verdict...")
            
            # --- DETAILED TRACE LOGGING (Decider) ---
            # Capture the exact prompt seen by the Decider (Analyst + Strategist + System Prompt)
            # Note: We pass the constructed user part (decider_context). 
            # The System Prompt is passed as argument 'system_prompt'.
            decider_full_input = {
                "system_prompt": system_prompt,
                "history_length": len(conversation_history),
                "injected_context": decider_context
            }
            
            final_decision_response, tool_call_str = self.main_agent.run_turn(
                system_prompt, conversation_history + [{"role": "user", "parts": [decider_context]}]
            )
            
            save_agent_trace(
                run_id=run_id,
                agent_name="Decider",
                system_prompt=system_prompt,
                inputs=decider_context, # Save the specific context we injected
                output=final_decision_response,
                metadata={"model": self.main_agent.model_name}
            )
            
            # --- 5. LOG FINAL VERDICT ---
            log_agent_artifact(run_id, "decider_verdict.md", final_decision_response)
            log(f"[Orchestrator] Cycle completed. Decision logged to MLFlow.")

            full_thought_process = f"""
{BLUE}<thought_step_1 - Factual Analysis>
(Analyst - Remote Worker)
{analysis}
</thought_step_1 - Factual Analysis>{RESET}
{MAGENTA}<thought_step_2 - Strategic Goal>
(Strategist - Remote Worker)
{hypothesis}
</thought_step_2 - Strategic Goal>{RESET}
{GREEN}<thought_step_3 - Final Decision>
(Decider)
{final_decision_response.split('<tool_code>')[0].strip()}
</thought_step_3 - Final Decision>{RESET}
"""
            return full_thought_process, tool_call_str

    async def launch_parallel_tools(self, tool_calls: List[Dict[str, Any]], thread_safe_state: Dict[str, Any], max_concurrent_gpu: int) -> List[Dict[str, Any]]:
        """
        Executes a list of tool calls in parallel, using a semaphore to manage exclusive GPU access.
        """
        log(f"[Parallel Runner] Received {len(tool_calls)} tool calls.")
        GPU_TOOLS = ["run_training_trial", "run_strategic_optuna_sweep", "run_supernet_search"]

        # Update resources if changed
        if max_concurrent_gpu != self.num_gpus:
            log(f"[ResourceAllocator] Updating resources from {self.num_gpus} to {max_concurrent_gpu} GPUs.")
            self.num_gpus = max_concurrent_gpu
            self.gpu_semaphore = threading.Semaphore(max_concurrent_gpu)
            with self.gpu_lock:
                self.gpu_pool = list(range(max_concurrent_gpu))
        
        gpu_tasks_to_create = []
        cpu_tasks_to_create = []
        
        for call in tool_calls:
            if call.get("tool_name") in GPU_TOOLS:
                gpu_tasks_to_create.append(call)
            else:
                cpu_tasks_to_create.append(call)

        log(f"[Parallel Runner] Categorized tasks: {len(gpu_tasks_to_create)} GPU, {len(cpu_tasks_to_create)} CPU.")

        # Create asyncio tasks
        tasks = []
        
        # --- Schedule CPU tasks ---
        for call in cpu_tasks_to_create:
            task = asyncio.create_task(
                asyncio.to_thread(self.execute_tool_func, call.get("tool_name"), call.get("args", {}), thread_safe_state)
            )
            tasks.append(task)
            
        # --- Schedule GPU tasks using the semaphore wrapper ---
        for call in gpu_tasks_to_create:
            task = asyncio.create_task(self._run_gpu_tool_wrapper(call, thread_safe_state))
            tasks.append(task)

        # --- Execute all scheduled tasks ---
        results = []
        if tasks:
            log(f"[Parallel Runner] Executing {len(tasks)} tasks concurrently...")
            try:
                # gather will wait for all tasks to complete
                parallel_results = await asyncio.gather(*tasks)
                results.extend(parallel_results)
            except Exception as e:
                log(f"[Parallel Runner] CRITICAL ERROR during asyncio.gather: {e}")
                results.append(
                    {"status": "error", "message": f"A critical error occurred during parallel execution: {e}"}
                )

        log(f"[Parallel Runner] Finished execution. Returning {len(results)} results.")
        # --- POST-SUPERNET SUB-AGENT HOOK ---
        # Automatically triggers model construction and optimization if Supernet found a genotype.
        for res in results:
            try:
                # Check if this result is from a Supernet search
                # Note: We rely on the tool output structure or look for "genotype" in the result
                result_data = res.get("result", {})
                if isinstance(result_data, str):
                    try:
                            result_data = json.loads(result_data)
                    except:
                            pass
                
                # [Refactor] PostSupernetAgent Hook Removed.
                pass

            except Exception as e:
                log(f"[PostSupernetAgent] Hook failed (legacy): {e}")

        return results





            

            

            

            

            









class InnovationCollaborator:
    """
    A specialized sub-agent that works WITH the main agent to:
    1. SCOUT: Propose specific architectures based on loose hypotheses.
    2. CODE: Write the actual Python code for the model.
    3. VALIDATE: Ensure the code constructs a valid Keras model.
    """

    def __init__(self, main_agent: ConversationalAgent, coding_model_name: str = None, coding_agent: ConversationalAgent = None):
        self.main_agent = main_agent
        self.coding_model_name = coding_model_name
        self.coding_agent = coding_agent
        self.execute_tool_func = None

    def _extract_python_code(self, response: str) -> str:
        tool_code_match = re.search(r"<tool_code>(.*?)</tool_code>", response, re.DOTALL)
        if tool_code_match:
            try:
                json_str = tool_code_match.group(1).strip()
                data = json.loads(json_str)
                if "args" in data and "code" in data["args"]:
                    return data["args"]["code"].strip()
            except (json.JSONDecodeError, KeyError):
                pass
        markdown_match = re.search(r"```python\n(.*?)```", response, re.DOTALL)
        if markdown_match:
            return markdown_match.group(1).strip()
        cleaned_response = re.sub(r"</?tool_code>", "", response)
        cleaned_response = re.sub(r"</?thought_step.*?>", "", cleaned_response)
        return cleaned_response.strip()

    def _run_sub_agent(self, system_prompt: str, conversation_history: list, agent_prompt: str) -> str:
        # Prio 1: Dedicated Coding Agent (Hybrid Provider Support)
        # Prio 1: Dedicated Coding Agent (Hybrid Provider Support)
        if self.coding_agent:
             log(f"[Innovation Team] Using Dedicated Coding Agent ({self.coding_agent.model_name})...")
             try:
                 response, _ = self.coding_agent.run_turn(
                    system_prompt, conversation_history + [{"role": "user", "parts": [agent_prompt]}]
                )
                 log(f"[Innovation Team] Coding Agent response length: {len(response)}")
                 return response
             except Exception as e:
                 log(f"[Innovation Team] ❌ Coding Agent FAILED: {e}")
                 # Fallback? No, if dedicated agent fails, we should know.
                 raise e

        # Prio 2: Main Agent with Model Override (Same Provider)
        log(f"[Innovation Team] Using Main Agent with override: {self.coding_model_name}...")
        response, _ = self.main_agent.run_turn(
            system_prompt, conversation_history + [{"role": "user", "parts": [agent_prompt]}],
            model_override=self.coding_model_name
        )
        return response

    def _get_best_architecture_code(self, recent_trials_data: Any) -> str:
        """
        Extracts the code of the best performing architecture from recent trials.
        Only considers trials that used a custom architecture file (.py).
        """
        best_code = ""
        best_acc = -1.0
        best_arch_name = "None"

        try:
            # Extract list from tool response if needed
            trials_list = []
            if isinstance(recent_trials_data, dict):
                trials_list = recent_trials_data.get("recent_trials", [])
            elif isinstance(recent_trials_data, list):
                trials_list = recent_trials_data
            
            if not trials_list:
                 return "No recent trials available to analyze."

            # Filter for trials with custom architecture files
            custom_arch_trials = [
                t for t in trials_list 
                if isinstance(t, dict) and t.get("architecture", "").endswith(".py") and "mean_accuracy" in t
            ]
            
            if not custom_arch_trials:
                return "No custom architecture files found in recent history."

            # Find the best one
            # handle potential mixed types or missing values robustly
            def get_acc(t):
                try: 
                    return float(t.get("mean_accuracy", 0.0))
                except: 
                    return 0.0

            best_trial = max(custom_arch_trials, key=get_acc)
            best_acc = get_acc(best_trial)
            best_arch_name = best_trial["architecture"]
            
            # Read the file
            # DEBUG PRINT
            # print(f"DEBUG: PERSISTENT_PATHS in agent: {PERSISTENT_PATHS}")
            arch_path = os.path.join(PERSISTENT_PATHS["custom_architectures_dir"], best_arch_name)
            # print(f"DEBUG: Checking path: {arch_path}, Exists: {os.path.exists(arch_path)}")

            if os.path.exists(arch_path):
                with open(arch_path, "r") as f:
                    best_code = f"# --- Best Architecture: {best_arch_name} (Acc: {best_acc:.4f}) ---\n"
                    best_code += f.read()
            else:
                best_code = f"Error: Best architecture file '{best_arch_name}' not found locally at {arch_path}. Paths: {PERSISTENT_PATHS}"

        except Exception as e:
            log(f"[Innovation Team] Error reading best architecture: {e}")
            best_code = f"Error retrieving architecture code: {e}"

        return best_code

    def _get_named_architecture_code(self, filename: str) -> str:
        """
        Read the full source of an explicit custom architecture .py file from the
        custom_architectures directory. Used by the upgrade (mutation) entry point to
        pin a specific evolutionary base instead of auto-selecting the best trial.
        Returns the source, or an 'Error: ...' string the caller checks before use.
        """
        try:
            if not filename or not filename.endswith(".py"):
                return f"Error: '{filename}' is not a valid .py architecture file."
            arch_path = os.path.join(PERSISTENT_PATHS["custom_architectures_dir"], filename)
            if not os.path.exists(arch_path):
                return f"Error: architecture file '{filename}' not found at {arch_path}."
            with open(arch_path, "r") as f:
                return f.read()
        except Exception as e:
            log(f"[Innovation Team] Error reading named architecture '{filename}': {e}")
            return f"Error retrieving '{filename}': {e}"

    def _run_pattern_scout(self, model_type: str, metadata_info: str) -> Tuple[str, str]:
        """
        Analyzes the Strategic Graph and Log to find successful motifs.
        Returns a tuple: (Design Strategy, Source of Insight)
        """
        log("[Innovation Team] Pattern Scout: Analyzing past successes...")
        
        # 1. Get Graph Data
        graph_data = self.execute_tool_func("get_strategic_graph", {})
        
        # 2. Get Strategic Log (High-level history)
        # We can't easily access the full log here without passing it, but we can use 'get_recent_trials'
        recent_trials_data = self.execute_tool_func("get_recent_trials", {"n": 10, "architecture_filter": model_type})
        
        # 3. Get Code of Best Architecture
        # NEW CAPABILITY: Read the actual code of the best model found so far
        best_architecture_code = self._get_best_architecture_code(recent_trials_data)

        scout_prompt = f"""
        **You are the Pattern Scout.** Your job is to analyze the provided data and identify "motifs" (successful architectural patterns) to guide the Architect.
        
        **Current Context:**
        - Model Type: {model_type}
        - Data Info: {metadata_info}
        
        **Strategic Graph Data (Nodes = States, Edges = Actions):**
        {json.dumps(graph_data, indent=2)[:2000]}... (truncated)
        
        **Recent Trials:**
        {json.dumps(recent_trials_data, indent=2)}

        **Best Architecture Code (Reference):**
        {best_architecture_code}
        
        **Task:**
        1. Identify what is working (e.g., "High dropout (>0.5) correlates with stability", "Residual connections are essential for deep 1D CNNs").
        2. Identify what is failing.
        3. Formulate a **Design Strategy** for the Architect.
        
        **Output Format:**
        <strategy>
        [Your detailed design strategy here]
        </strategy>
        <confidence>
        [High/Medium/Low]
        </confidence>
        """
        
        response = self._run_sub_agent("You are an expert Data Scientist specializing in Neural Architecture Search.", [], scout_prompt)
        
        strategy_match = re.search(r"<strategy>(.*?)</strategy>", response, re.DOTALL)
        confidence_match = re.search(r"<confidence>(.*?)</confidence>", response, re.DOTALL)
        
        strategy = strategy_match.group(1).strip() if strategy_match else "Focus on standard best practices."
        confidence = confidence_match.group(1).strip().lower() if confidence_match else "low"
        
        return strategy, confidence, best_architecture_code

    def _run_literature_check(self, query: str) -> str:
        """
        Runs a web research task to find SOTA architectures.
        """
        log(f"[Innovation Team] Literature Check: Searching for '{query}'...")
        result = self.execute_tool_func("run_web_research", {"query": query, "max_sites": 3})
        if result.get("status") == "completed":
            return result.get("report", "No report generated.")
        return "Literature check failed."

    def _extract_motifs(self, code: str, accuracy: float) -> str:
        """
        Analyzes successful code to extract reusable motifs.
        """
        motif_prompt = f"""
        **You are the Motif Extractor.**
        The following code achieved an accuracy of {accuracy:.4f}.
        Extract the key architectural "motif" (e.g., a specific block structure, a unique layer configuration) that likely contributed to this success.
        
        **Code:**
        ```python
        {code}
        ```
        
        **Output:**
        A concise description of the motif (max 2 sentences).
        """
        response = self._run_sub_agent("You are an expert Deep Learning Researcher.", [], motif_prompt)
        return response.strip()


    def run_innovation_cycle(
        self,
        system_prompt: str,
        conversation_history: list,
        initial_hypothesis: str,
        architecture_context: str,  # This is the FALLBACK
        execute_tool_func: callable,
        model_type: str,
        metadata_info: str = None,
        thread_safe_state: dict = None,
        base_file: str = None,
    ) -> tuple[str, str | None]:
        self.execute_tool_func = execute_tool_func
        self._thread_safe_state = thread_safe_state  # stored for _find_or_create_compatible_manifest
        log("[Innovation Team] Cycle initiated. Goal: Create a working architecture.")

        # --- 1. PATTERN SCOUT & LITERATURE CHECK (New Step) ---
        design_strategy, confidence, best_architecture_code = self._run_pattern_scout(model_type, metadata_info)
        log(f"[Innovation Team] Pattern Scout Strategy: {design_strategy} (Confidence: {confidence})")
        
        sota_report = ""
        supernet_genotype = ""

        # --- NEW: SUPERNET SEARCH INTEGRATION ---
        # If confidence is low OR explicitly requested (we can add a heuristic here later), try NAS.
        # For now, we trigger it if the design strategy explicitly mentions "NAS" or "search", 
        # or if we add a 'use_supernet' flag to the method (which we will assume is passed in kwargs if we update the signature).
        
        # Heuristic: If confidence is low, maybe NAS can help find a baseline?
        # Or if the user asked for it.
        
        # Let's check if we can infer 'use_supernet' from the context or just try it if confidence is low.
        # For this implementation, I'll add a heuristic: if "NAS" or "Supernet" is in the initial_hypothesis.
        use_supernet = "supernet" in initial_hypothesis.lower() or "nas" in initial_hypothesis.lower()
        
        # --- NEW: STRATEGIC SAFEGUARD (BASELINE FIRST) ---
        # Do not allow auto-trigger if we are already doing great with vanilla architectures.
        # Check thread_safe_state for 'best_accuracy'
        
        # We need to access thread_safe_state. It is not passed explicitly to _run_pattern_scout...
        # Wait, _run_pattern_scout is called by propose_intelligent_architecture.
        # The 'self.execute_tool_func' can enable us to peek at state if we use a special tool, 
        # or we can rely on the fact that InnovationCollaborator is usually call FROM a tool that has state access.
        
        # Simpler approach: If "Supernet" is requested but we have high accuracy, we LOG A WARNING but might still proceed 
        # if the user REALLY wants it. But for "auto" suggestions, we should kill it.
        
        # However, here 'use_supernet' is determined by KEYWORDS in the hypothesis.
        # If the Strategist wrote "Supernet", it probably means it wants it.
        # But we changed the Strategist prompt to NOT ask for it unless failing.
        
        # So, the safeguard here is for when the Strategist hallucinates or the user asks for it prematurely?
        # Let's add a "Smart Override".
        
        # We can get 'optimization_status' to check best accuracy.
        try:
             opt_status = self.execute_tool_func("get_optimization_status", {})
             current_best_acc = opt_status.get("best_accuracy", 0.0)
             if current_best_acc > 0.95 and use_supernet:
                 log(f"[Innovation Team] ⚠️ Request for Supernet detected, but accuracy is already high ({current_best_acc:.4f}).")
                 log(f"[Innovation Team] Proceeding with caution. (Ideally we should stick to optimization).")
                 # Optional: force disable? 
                 # use_supernet = False 
                 # Let's keep it as a soft warning for now, trusting the Strategist changes to reduce the attempts.
        except:
             pass
        
        if use_supernet:
             # --- REUSABILITY CHECK ---
             # Check if we were handed an existing Genotype file (e.g. from the Supernet Routine)
             # Format usually: {representation}_Genotype_{timestamp}.py
             is_genotype_file = architecture_context and "Genotype" in architecture_context and architecture_context.endswith(".py")
             
             if is_genotype_file:
                 log(f"[Innovation Team] ⏩ Skipping new Supernet search. Context implies existing genotype: {architecture_context}")
                 design_strategy += f"\n\n**EXISTING GENOTYPE:** The user provided an existing Supernet-derived backbone: '{architecture_context}'. \nYou MUST use this file's structure as the base for your refinement. Do NOT reinvent the wheel."
                 # inferred_accuracy = 0.90 # We don't know it exactly without parsing, but the Routine implied it's decent.
             else:
                 log("[Innovation Team] Supernet Search triggered by hypothesis keywords.")
                 # We need a manifest.
                 # Try to guess representation from context
                 rep_type = architecture_context
                 manifest_name = self._find_or_create_compatible_manifest(rep_type, thread_safe_state=self._thread_safe_state)
                 
                 if manifest_name:
                     log(f"[Innovation Team] Launching Supernet Search on '{manifest_name}'...")
                     search_result = self.execute_tool_func("run_supernet_search", {
                         "dataset_id": manifest_name,
                         "representation_type": rep_type,
                         "max_epochs": 150 # Increased for better validation
                     })
                     
                     if search_result.get("status") == "completed":
                         supernet_genotype = search_result.get("genotype", "")
                         supernet_accuracy = search_result.get("accuracy", 0.0)
                         log(f"[Innovation Team] Supernet found genotype: {supernet_genotype} (Acc: {supernet_accuracy:.3f})")
                         
                         # --- EARLY REJECTION LOGIC ---
                         if supernet_accuracy < 0.40: # Threshold for "promising" (can be tuned)
                             log(f"[Innovation Team] ❌ Supernet accuracy ({supernet_accuracy:.3f}) is too low. Aborting cycle.")
                             final_thoughts = f"<thought_step_failure>Supernet search indicated low potential ({supernet_accuracy:.3f}) for this hypothesis. Aborting to save resources.</thought_step_failure>"
                             return final_thoughts, None
                         
                         design_strategy += f"\n\n**NAS Insight:** The One-Shot NAS supernet suggests the following optimal cell structure (Genotype): {supernet_genotype}. \n**Validation Accuracy:** {supernet_accuracy:.3f}. \nYou MUST use this structure in your design."
                     else:
                         log(f"[Innovation Team] Supernet search failed: {search_result.get('message')}")

        if "low" in confidence or "medium" in confidence:
            log("[Innovation Team] Confidence is not High. Triggering Literature Check...")
            sota_report = self._run_literature_check(f"SOTA {model_type} architectures for {metadata_info}")
            
        # --- 2. ARCHITECT'S TURN (Conception with Evolution) ---
        
        # --- ARCHITECTURE CONTEXT PREPARATION ---
        library_path = os.path.join(os.path.dirname(__file__), "cluster", "library", "architectures")

        # 1. Determine dimensionality tag to filter relevant library files
        repr_tag = "1d"
        if "2D" in model_type or "IMAGE" in model_type:
            repr_tag = "2d"
        elif "3D" in model_type or "VIDEO" in model_type:
            repr_tag = "3d"

        # 2. Scan library and identify relevant files for this dimensionality
        available_files: list[str] = []
        relevant_files: list[str] = []
        library_catalogue = "unavailable"
        library_relevant = "unavailable"
        library_previews_text = "No library files found."
        try:
            available_files = sorted([f for f in os.listdir(library_path) if f.endswith(".py")])
            relevant_files = sorted([f for f in available_files if f"_{repr_tag}.py" in f])
            library_catalogue = ", ".join(available_files) if available_files else "None"
            library_relevant = ", ".join(relevant_files) if relevant_files else "None"

            # Build short preview blocks (first 20 lines each) for all relevant files
            preview_blocks = []
            for fname in relevant_files:
                fpath = os.path.join(library_path, fname)
                try:
                    with open(fpath, "r") as fh:
                        preview = "".join(fh.readlines()[:20]).strip()
                    preview_blocks.append(f"### `{fname}`\n```python\n{preview}\n... (truncated)\n```")
                except Exception as fe:
                    preview_blocks.append(f"### `{fname}`\n# Could not read: {fe}")
            library_previews_text = "\n\n".join(preview_blocks) if preview_blocks else "No relevant library files."
        except Exception as e:
            log(f"[Innovation Team] Architecture Library scan failed: {e}")

        # 3. Extract the CURRENT BASE MODEL source from model_factory.py
        # This is always clean, always available, and the correct evolutionary starting point.
        base_model_source = ""
        try:
            import inspect
            import importlib
            import model_factory as _mf
            importlib.reload(_mf)
            builder_fn = _mf.MODEL_BUILDERS.get(model_type)
            if builder_fn and callable(builder_fn):
                base_model_source = inspect.getsource(builder_fn)
                log(f"[Innovation Team] Extracted base model source: {model_type} ({builder_fn.__name__})")
            else:
                base_model_source = f"# No base builder registered for model_type='{model_type}' in model_factory.MODEL_BUILDERS."
        except Exception as e:
            base_model_source = f"# Could not extract base model source: {e}"
            log(f"[Innovation Team] Warning: Base model source extraction failed: {e}")

        # --- UPGRADE MODE: pin an explicit custom architecture file as the base ---
        # When base_file is supplied (mutation entry point), SECTION 1 of the Architect
        # prompt must show THAT file's source, not the standard model_factory builder, so
        # the LLM evolves the pinned architecture instead of the generic baseline.
        if base_file:
            pinned_source = self._get_named_architecture_code(base_file)
            if pinned_source and not pinned_source.startswith("Error"):
                base_model_source = pinned_source
                design_strategy = (
                    f"**UPGRADE TARGET — `{base_file}`:** You are evolving this existing, working "
                    f"architecture, NOT writing one from scratch. SECTION 1 below is its actual source. "
                    f"Apply the hypothesis as a TARGETED modification: preserve every component that "
                    f"already works and change only what the hypothesis requires.\n\n" + (design_strategy or "")
                )
                log(f"[Innovation Team] UPGRADE MODE: pinned base architecture '{base_file}'.")
            else:
                log(f"[Innovation Team] UPGRADE MODE: could not read '{base_file}' "
                    f"({pinned_source[:80]}). Falling back to standard base source.")

        current_manifest = "UNKNOWN"
        if thread_safe_state:
            # 1. First, check direct state variable used by the rest of the system
            if "selected_raw_data_source" in thread_safe_state and thread_safe_state["selected_raw_data_source"]:
                current_manifest = thread_safe_state["selected_raw_data_source"]
            # 2. Fallback to extracting from config
            elif "cfg" in thread_safe_state:
                cfg = thread_safe_state.get("cfg", {})
                if hasattr(cfg, "get"):
                    # Handle both native dict and OmegaConf DictConfig
                    data_cfg = cfg.get("data", {})
                    if hasattr(data_cfg, "get"):
                        current_manifest = data_cfg.get("raw_data_source", "UNKNOWN")

        architect_prompt = f"""
            **You are part of the Innovation team and you are the Architect.** Your task is to write the Python code for a new model.
            
            **Input Context:**
            - **Hypothesis:** {initial_hypothesis}
            - **Design Strategy (from Pattern Scout):** {design_strategy}
            - **SOTA Report (from Literature):** {sota_report}

            ---
            ## SECTION 1 — CURRENT BASE MODEL (Your Primary Starting Point)
            This is the exact source code of the model currently being optimized (`{model_type}`).
            **You MUST start from this code** unless the hypothesis requires a fundamentally different architecture.
            ```python
            {base_model_source}
            ```

            ---
            ## SECTION 2 — ADVANCED LIBRARY TEMPLATES (Optional Upgrade Scaffolds)
            The following pre-built architectures are available.
            **Only use one of these if your hypothesis explicitly calls for it** (e.g. ResNet, DenseNet, Transformer).
            If you choose a library template instead of Section 1, declare it with: `<library_choice>filename.py</library_choice>`
            
            Available for `{repr_tag}` tasks: `{library_relevant}`
            Full library: `{library_catalogue}`
            
            {library_previews_text}

            ---
            **Instructions:**
            1.  **Iterative Evolution (CRITICAL):**
                - **DEFAULT:** Start from **SECTION 1 (Current Base Model)** and evolve it per the hypothesis.
                - **EXCEPTION:** If the hypothesis requires a fundamentally different architecture (e.g. ResNet, Transformer),
                  pick the best match from **SECTION 2** and declare it with `<library_choice>filename.py</library_choice>`.
                - **DO NOT** rewrite from scratch unless the hypothesis explicitly asks for a "clean slate".
            
            2.  **Filename:** Propose a short, descriptive, snake_case filename. Enclose it in `<filename>...</filename>` tags.
            
            3.  **Representation (MANDATORY Strict Enforcement: 100%)**
                **YOU MUST COPY-PASTE ONE OF THESE EXACT VALUES. DO NOT INVENT NEW TYPES.**
                Valid options (choose ONE and copy EXACTLY):
                - '1D_CNN'
                - '2D_SPECTROGRAM'
                - '2D_GAF'
                - '2D_CWT_SCALOGRAM'
                - '2D_IMAGE'
                - '3D_VIDEO'
                - '3D_GAF_VIDEO'       # spectrum split into N segments → GAF per segment (local structure)
                - '3D_DYNAMIC_GAF'    # full spectrum at N Gaussian σ levels 5.0→0.0 → GAF per σ (multi-scale focus)

                **FORBIDDEN:** Do NOT use '2D_HYBRID', '2D_CUSTOM', '2D_ADVANCED', or ANY other value not listed above.
                **FORMAT:** Enclose your choice in `<representation>EXACT_VALUE_FROM_LIST</representation>` tags.
            
            4.  **Manifest (CRITICAL):** You MUST use the exact dataset manifest currently configured: `{current_manifest}`.
                Enclose it EXACTLY in `<manifest>{current_manifest}</manifest>` tags. DO NOT INVENT A NEW NAME.
            5.  **Code:** Write the complete Python code for the `build_model` function.
                - Signature MUST be `build_model(input_shape, num_classes, params: dict)`.
                - All imports MUST be inside the function.
            """

        


        raw_architect_response = self._run_sub_agent(system_prompt, conversation_history, architect_prompt)

        filename_match = re.search(r"<filename>(.*?)</filename>", raw_architect_response)
        if filename_match:
            filename = filename_match.group(1).strip()
            filename = re.sub(r"[^a-zA-Z0.9_-]", "", filename)
            if not filename.endswith(".py"):
                filename += ".py"
        else:
            filename = f"arch_fallback_{abs(hash(initial_hypothesis)) % 10000}.py"
        log(f"[Innovation Team] Architect proposed filename: '{filename}'")

        # --- START FIX: Extract the representation type ---
        representation_match = re.search(
            r"<representation>(.*?)</representation>", raw_architect_response, re.IGNORECASE
        )
        if representation_match:
            # 1. Get the representation from the Architect
            representation_type_for_trial = representation_match.group(1).strip()
            log(f"[Innovation Team] Architect specified representation: '{representation_type_for_trial}'")
        else:
            # 2. Fallback to the context if the Architect forgot
            log(
                f"[Innovation Team] WARNING: Architect did not specify <representation>. Falling back to context: '{architecture_context}'"
            )
            representation_type_for_trial = architecture_context
        # --- END FIX ---

        # --- START DYNAMIC MANIFEST LOGIC ---
        manifest_match = re.search(r"<manifest>(.*?)</manifest>", raw_architect_response, re.IGNORECASE)
        if manifest_match:
            manifest_name_for_trial = manifest_match.group(1).strip()
            log(f"[Innovation Team] Architect specified manifest: '{manifest_name_for_trial}'")
        else:
            log(
                "[Innovation Team] WARNING: Architect did not specify <manifest>. Attempting to find or create a compatible one."
            )
            manifest_name_for_trial = self._find_or_create_compatible_manifest(representation_type_for_trial, thread_safe_state=self._thread_safe_state)
            if manifest_name_for_trial is None:
                log(
                    "[Innovation Team] CRITICAL: Failed to find or create a compatible manifest. Aborting innovation cycle."
                )
                final_thoughts = "<thought_step_failure>Could not find or create a compatible data manifest. The innovation cycle cannot proceed without valid data configuration.</thought_step_failure>"
                return final_thoughts, None
        # --- END DYNAMIC MANIFEST LOGIC ---

        code_to_test = self._extract_python_code(raw_architect_response)
        log("[Innovation Team] Architect drafted the initial code.")

        # --- 2. ORCHESTRATOR WRITES V1 AND CREATES CHANGELOG ---
        final_thoughts = f"<thought_step_1 - Conception>\n**Hypothesis:** {initial_hypothesis}\n**Proposed Filename:** {filename}\n**Architect's Draft:**\n```python\n{code_to_test}\n```\n</thought_step_1 - Conception>\n"

        self.execute_tool_func("write_architecture_file", {"filename": filename, "code": code_to_test})

        changelog_entry = f"# Changelog for {filename}\n\n## Version 1.0 (Attempt 1)\n- Initial creation based on hypothesis: {initial_hypothesis}\n"
        self.execute_tool_func(
            "append_to_architecture_changelog", {"py_filename": filename, "log_entry": changelog_entry}
        )

        max_retries = 6
        previous_error = None  # Circuit breaker: track last error
        repeated_error_count = 0
        for i in range(max_retries):
            log(f"[Innovation Team] --- Forge Attempt {i+1}/{max_retries} for '{filename}' ---")

            try:
                # --- 3. VALIDATION STEP ---
                # --- START FIX: Use the new representation variable ---
                if "1D" in representation_type_for_trial:
                    expected_type = "1D"
                elif "3D" in representation_type_for_trial:
                    expected_type = "3D"
                else:
                    expected_type = "2D"  # Default to 2D for 2D_GAF, 2D_SPECTROGRAM etc.

                log(
                    f"[Innovation Team] Inferred expected_input_type '{expected_type}' from representation '{representation_type_for_trial}'"
                )
                # --- END FIX ---

                validation_result = self.execute_tool_func(
                    "validate_architecture_file", {
                        "filename": filename, 
                        "expected_input_type": expected_type,
                        "context_reason": None
                    }
                )

                if validation_result.get("status") == "error":
                    log("[Innovation Team] ❌ Syntax validation failed. Activating Debugger.")
                    error_message = validation_result.get("traceback", "Unknown validation error.")
                    
                    # CIRCUIT BREAKER: Check if this is the same error as before
                    error_signature = error_message[:200] if error_message else ""  # First 200 chars as signature
                    if error_signature == previous_error:
                        repeated_error_count += 1
                        log(f"[Innovation Team] ⚠️ SAME ERROR REPEATED ({repeated_error_count}/3)")
                        if repeated_error_count >= 3:
                            log("[Innovation Team] 🛑 CIRCUIT BREAKER ACTIVATED: Same error 3 times. Aborting innovation cycle.")
                            final_thoughts += f"<thought_step_failure>Circuit breaker activated after {i+1} attempts. The same validation error repeated 3 times, indicating a systemic issue that cannot be auto-fixed. Manual intervention required.</thought_step_failure>\n"
                            return final_thoughts, None
                    else:
                        previous_error = error_signature
                        repeated_error_count = 1  # Reset counter for new error
                    
                    final_thoughts += f"<thought_step_2a - Debug Syntax>\n**Attempt {i+1} failed validation.**\n**Error:** {error_message}\n"

                    # --- 4. DEBUGGER STEP (with file system memory) ---
                    # Log the error to the changelog *before* calling the debugger
                    log_entry = f"\n\n## FAILED VALIDATION (Attempt {i+1})\n**Error:**\n```\n{error_message}\n```\n"
                    self.execute_tool_func(
                        "append_to_architecture_changelog", {"py_filename": filename, "log_entry": log_entry}
                    )

                    # Read the files to get the current state
                    code_read_result = self.execute_tool_func("read_file_from_architectures", {"filename": filename})
                    changelog_read_result = self.execute_tool_func(
                        "read_file_from_architectures", {"filename": filename.replace(".py", ".md")}
                    )

                    # Call the Debugger with the full context
                    debug_response = self._debug_code(  # Also, capture the full response
                        system_prompt,
                        conversation_history,
                        code_read_result.get("content", ""),  # Pass current code
                        error_message,
                        "Syntax",  # Error type
                        changelog_read_result.get("content", ""),  # Pass changelog
                        current_representation=representation_type_for_trial,
                    )

                    if "#CREATE_NEW_MANIFEST" in debug_response:
                        log("[Innovation Team] Debugger requested a new manifest. Creating one...")
                        manifest_name_for_trial = self._find_or_create_compatible_manifest(
                            representation_type_for_trial, thread_safe_state=self._thread_safe_state
                        )
                        log(f"[Innovation Team] Using new manifest: '{manifest_name_for_trial}'")

                    # Also extract the code from the response
                    code_to_test = self._extract_python_code(debug_response)

                    # Write the debugger's fix to disk and log the attempt
                    self.execute_tool_func("write_architecture_file", {"filename": filename, "code": code_to_test})
                    log_entry = (
                        f"\n\n## DEBUG FIX APPLIED (Attempt {i+1})\n- Debugger applied a new fix. Re-validating...\n"
                    )
                    self.execute_tool_func(
                        "append_to_architecture_changelog", {"py_filename": filename, "log_entry": log_entry}
                    )

                    final_thoughts += (
                        f"**Debugger's Fix:**\n```python\n{code_to_test}\n```\n</thought_step_2a - Debug Syntax>\n"
                    )
                    continue  # Go to the next loop iteration to re-validate

                log("[Innovation Team] ✅ Syntax validation successful.")
                log_entry = f"\n\n## PASSED VALIDATION (Attempt {i+1})\n- Code is syntactically valid. Proceeding to training trial.\n"
                self.execute_tool_func(
                    "append_to_architecture_changelog", {"py_filename": filename, "log_entry": log_entry}
                )

                # --- 5. TRAINING TRIAL STEP (smoke test: few epochs to verify viability) ---
                trial_args = {
                    "manifest_name": manifest_name_for_trial,
                    "custom_architecture_file": filename,
                    "representation_type": representation_type_for_trial,
                    "epochs": 10,
                    "patience": 5,
                }
                trial_result = self.execute_tool_func("run_training_trial", trial_args)

                if trial_result.get("status") == "completed":
                    log(f"[Innovation Team] ✅✅✅ Trial completed successfully! Architecture '{filename}' is viable.")

                    # --- NEW: Motif Extraction ---
                    try:
                        accuracy = trial_result.get("mean_accuracy", 0.0)
                        if accuracy > 0.3:  # Memorize any viable result (SERS baseline ~0.56)
                            motif = self._extract_motifs(code_to_test, accuracy)
                            log(f"[Innovation Team] Extracted Motif: {motif}")
                            self.execute_tool_func("memorize_finding", {
                                "finding": f"Successful Motif for {model_type} (Acc: {accuracy:.3f}): {motif}",
                                "experiment_name": filename
                            })
                    except Exception as e:
                        log(f"[Innovation Team] Warning: Motif extraction failed: {e}")
                    # -----------------------------
                    log_entry = f"\n\n## SUCCESS (Attempt {i+1})\n- Architecture passed validation and training.\n- **Final Result:** {json.dumps(trial_result)}\n"
                    self.execute_tool_func(
                        "append_to_architecture_changelog", {"py_filename": filename, "log_entry": log_entry}
                    )

                    final_thoughts += f"<thought_step_3 - Success>\nThe architecture passed validation and completed a training trial successfully.\n**Final Trial Result:**\n{json.dumps(trial_result)}\n</thought_step_3 - Success>\n"
                    return final_thoughts, {
                        "architecture_file": filename,
                        "representation_type": representation_type_for_trial,
                        "manifest_name": manifest_name_for_trial,
                    }

                # --- NEW: Check for System/Data Errors ---
                error_message = trial_result.get("message", "Unknown training error.")
                if "[SYSTEM_ERROR]" in error_message or ("Manifest" in error_message and "failed" in error_message):
                    log(f"[Innovation Team] 🛑 System/Data Error detected: {error_message}")
                    log("[Innovation Team] Aborting innovation cycle to prevent infinite architecture debugging loop.")
                    final_thoughts += f"<thought_step_2b - System Error>\n**Trial failed due to system/data error, not architecture.**\n**Error:** {error_message}\nAborting to prevent infinite loop.</thought_step_2b - System Error>\n"
                    return final_thoughts, None

                # --- 6. DEBUGGER STEP (Runtime Error) ---
                log("[Innovation Team] ❌ Training trial failed. Activating Debugger for runtime error.")
                error_message = trial_result.get("message", "Unknown training error.")
                final_thoughts += f"<thought_step_2b - Debug Runtime>\n**Attempt {i+1} failed during training.**\n**Error:** {error_message}\n"

                # Log the error
                log_entry = f"\n\n## FAILED TRAINING (Attempt {i+1})\n**Error:**\n```\n{error_message}\n```\n"
                self.execute_tool_func(
                    "append_to_architecture_changelog", {"py_filename": filename, "log_entry": log_entry}
                )

                # Read files for context
                code_read_result = self.execute_tool_func("read_file_from_architectures", {"filename": filename})
                changelog_read_result = self.execute_tool_func(
                    "read_file_from_architectures", {"filename": filename.replace(".py", ".md")}
                )

                # Call Debugger
                code_to_test = self._debug_code(
                    system_prompt,
                    conversation_history,
                    code_read_result.get("content", ""),
                    error_message,
                    "Runtime",
                    changelog_read_result.get("content", ""),
                    current_representation=representation_type_for_trial,
                )

                # Write the fix and log it
                self.execute_tool_func("write_architecture_file", {"filename": filename, "code": code_to_test})
                log_entry = f"\n\n## DEBUG FIX APPLIED (Attempt {i+1})\n- Debugger applied a new fix for runtime error. Re-validating...\n"
                self.execute_tool_func(
                    "append_to_architecture_changelog", {"py_filename": filename, "log_entry": log_entry}
                )

                final_thoughts += (
                    f"**Debugger's Fix:**\n```python\n{code_to_test}\n```\n</thought_step_2b - Debug Runtime>\n"
                )
                # Continue to the next loop iteration

            except Exception as e:
                log(f"[Innovation Team] CRITICAL ERROR during innovation cycle: {e}")
                final_thoughts += f"<thought_step_critical_error>An unexpected error occurred: {traceback.format_exc()}</thought_step_critical_error>"
                # Log the critical error
                log_entry = f"\n\n## CRITICAL ORCHESTRATOR ERROR (Attempt {i+1})\n**Error:**\n```\n{traceback.format_exc()}\n```\n"
                self.execute_tool_func(
                    "append_to_architecture_changelog", {"py_filename": filename, "log_entry": log_entry}
                )
                return final_thoughts, None

        log("[Innovation Team] ❌ Failed to create a working architecture after all retries.")
        log_entry = f"\n\n## FINAL FAILURE\n- The Innovation Team could not produce a working architecture after {max_retries} attempts.\n"
        self.execute_tool_func("append_to_architecture_changelog", {"py_filename": filename, "log_entry": log_entry})
        final_thoughts += "<thought_step_failure>The Innovation Team could not produce a working architecture after multiple attempts.</thought_step_failure>"
        return final_thoughts, None

    # In agent.py, replace the entire _debug_code function

    def _debug_code(
        self,
        system_prompt,
        conversation_history,
        broken_code,
        error_message,
        error_type,
        changelog_content: str,
        current_representation: str,
    ):
        """
        Implements the 'Council of Debuggers' strategy.
        1. Spawns 5 parallel Debugger Agents to propose fixes.
        2. Uses an Auditor Agent to review all proposals and select the best one.
        """
        log("[Council of Debuggers] 🏛️ Convening the Council...")

        # --- 1. DEFINE THE DEBUGGER PROMPT ---
        debugger_prompt_template = """
        **You are a member of the Council of Debuggers (Expert Keras/TensorFlow Engineer).**
        Your task is to analyze the error and propose a robust fix.
        
        **Context:**
        - **Representation:** '{current_representation}' (Ensure the code expects this input shape).
        - **Error Type:** {error_type}
        
        **History (Changelog):**
        ```markdown
        {changelog_content}
        ```

        **Error Message:**
        ```
        {error_message}
        ```

        **Broken Code:**
        ```python
        {broken_code}
        ```

        **Instructions:**
        1. Analyze the error and the code.
        2. **=Representation Fix (MANDATORY Strict Enforcement: 100%)**
           If the error message says "Invalid representation_type", you MUST change it to one of these EXACT values:
           - '1D_CNN'
           - '2D_SPECTROGRAM'
           - '2D_GAF'
           - '2D_CWT_SCALOGRAM'
           - '2D_IMAGE'
           - '3D_VIDEO'
           - '3D_GAF_VIDEO'       # spectrum split into N segments → GAF per segment (local structure)
           - '3D_DYNAMIC_GAF'    # full spectrum at N Gaussian σ levels 5.0→0.0 → GAF per σ (multi-scale focus)

           **DO NOT** suggest '2D_HYBRID', '2D_CUSTOM', or any value not in the list above.
           Use `<representation>EXACT_VALUE_FROM_LIST</representation>` to specify the corrected type.
        
        3. If a new manifest is needed, output `#CREATE_NEW_MANIFEST`.
        4. **CRITICAL:** Output the COMPLETE, FIXED `build_model` function.
        """

        # --- 2. PARALLEL EXECUTION OF 5 DEBUGGERS ---
        # FIX: Throttled to 2 to prevent "Split-Brain" and API Rate Limits (Production Stabilization Phase 1)
        num_debuggers = 2
        proposals = []

        def run_single_debugger(index):
            log(f"[Council] Debugger {index+1} is analyzing...")
            
            # Formatting the prompt with the specific context
            formatted_prompt = debugger_prompt_template.replace("{current_representation}", str(current_representation)) \
                                             .replace("{error_type}", str(error_type)) \
                                             .replace("{changelog_content}", str(changelog_content)) \
                                             .replace("{error_message}", str(error_message)) \
                                             .replace("{broken_code}", str(broken_code))
            
            response = self._run_sub_agent(
                f"You are Debugger #{index+1} of the Council.", 
                conversation_history, 
                formatted_prompt
            )
            
            code = self._extract_python_code(response)
            return {
                "id": index + 1,
                "response": response,
                "code": code
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_debuggers) as executor:
            futures = [executor.submit(run_single_debugger, i) for i in range(num_debuggers)]
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result["code"]: # Only accept if valid code was extracted
                        proposals.append(result)
                        log(f"[Council] Debugger {result['id']} submitted a proposal.")
                    else:
                        log(f"[Council] Debugger {result['id']} failed to produce code.")
                except Exception as e:
                    log(f"[Council] Error in debugger thread: {e}")

        if not proposals:
            log("[Council] ❌ CRITICAL: No debuggers produced valid code. Falling back to simple retry.")
            return broken_code 

        # --- 3. THE AUDITOR (DECISION MAKER) ---
        log(f"[Council] 🧑‍⚖️ The Auditor is reviewing {len(proposals)} proposals...")
        
        auditor_prompt = f"""
        **You are the Chief Auditor of the Innovation Council.**
        We have a broken Keras model and {len(proposals)} proposed fixes from the debuggers.
        Your task is to select the BEST solution and output the final code.

        **Original Error:**
        {error_message}

        **Proposals:**
        """
        
        for p in proposals:
            auditor_prompt += f"\n--- Proposal {p['id']} ---\n"
            auditor_prompt += f"```python\n{p['code']}\n```\n"

        auditor_prompt += """
        **Instructions:**
        1. Compare the proposals. Look for the one that best addresses the specific error (e.g., shape mismatch, API misuse).
        2. Select the most robust and correct code.
        3. **Output ONLY the final, complete Python code for `build_model`.**
        """

        auditor_response = self._run_sub_agent(
            "You are the Chief Auditor. Select the best fix.",
            conversation_history,
            auditor_prompt
        )
        
        final_code = self._extract_python_code(auditor_response)
        return final_code


    def _find_or_create_compatible_manifest(self, representation_type: str, thread_safe_state: dict = None) -> str:
        """
        Finds a compatible manifest or creates a new one if none exists.

        Search order:
          Priority 1a: generated_representations metadata contains representation_type
          Priority 1b: physical folder exists on disk (metadata may be stale)
          Priority 2:  recommended_architecture matches
          Priority 3:  manifest name contains representation_type string
          Fallback:    create new manifest cloning params from current active manifest
                       then call generate_representation to pre-process the data.
        """
        from state_manager import PERSISTENT_PATHS as _PP
        generated_datasets_dir = _PP.get("generated_datasets_dir", "processed_data/generated_datasets")

        log(f"[Innovation Team] Searching for a manifest compatible with '{representation_type}'...")

        # 1. List available datasets
        available_datasets_result = self.execute_tool_func("list_available_datasets", {})
        datasets_meta = available_datasets_result.get("datasets", {})

        # 2. Find a compatible manifest using Smart Layer Priority
        for manifest_name, metadata in datasets_meta.items():
            if not isinstance(metadata, dict):
                continue

            # Priority 1a: Representation already generated (metadata)
            generated = metadata.get("generated_representations", {})
            if representation_type in generated:
                log(f"[Innovation Team] Found perfectly compatible manifest '{manifest_name}' (generated_representations).")
                return manifest_name

            # Priority 1b: Physical folder exists on disk (A3 — stale metadata guard)
            phys_path = os.path.join(generated_datasets_dir, manifest_name, representation_type)
            if os.path.isdir(phys_path) and any(
                f.endswith(".npy") or f.endswith(".npz")
                for f in os.listdir(phys_path)
            ):
                log(f"[Innovation Team] Found compatible manifest '{manifest_name}' via physical disk check.")
                return manifest_name

            # Priority 2: Recommended architecture matches
            recommended = metadata.get("recommended_architecture", "")
            if isinstance(recommended, str) and recommended and representation_type.lower() in recommended.lower():
                log(f"[Innovation Team] Found compatible manifest '{manifest_name}' via recommended architecture.")
                return manifest_name

            # Priority 3: Fallback loose string match on manifest name
            if representation_type.lower() in manifest_name.lower():
                log(f"[Innovation Team] Found compatible manifest '{manifest_name}' via name string matching.")
                return manifest_name

        # -----------------------------------------------------------------------
        # No match found — create a new manifest (A1: clone params from current)
        # -----------------------------------------------------------------------
        log("[Innovation Team] No compatible manifest found. Creating a new one.")
        new_manifest_name = f"new_{representation_type.lower().replace('/', '_')}_manifest"

        # A1: Identify current active manifest and clone its base params
        base_params = {}
        try:
            # Find the manifest used by the most recent trial
            current_manifest_name = None
            if thread_safe_state:
                results_log = thread_safe_state.get("results_log", [])
                if results_log:
                    current_manifest_name = results_log[-1].get("manifest_name")
            if current_manifest_name and current_manifest_name in datasets_meta:
                base_params = dict(datasets_meta[current_manifest_name].get("params", {}))
                # Remove keys that should not carry over
                base_params.pop("description", None)
                log(f"[Innovation Team] Cloning base params from manifest '{current_manifest_name}': {list(base_params.keys())}")
        except Exception as e:
            log(f"[Innovation Team] WARNING: Could not clone base params: {e}. Using defaults.")

        # Override representation_type and add description
        base_params["representation_type"] = representation_type
        base_params["description"] = (
            f"Auto-created manifest for {representation_type} representation, "
            f"cloned from current dataset params."
        )

        creation_args = {
            "manifest_name": new_manifest_name,
            "params": base_params,
        }

        create_result = self.execute_tool_func("create_dataset_manifest", creation_args)

        # Honor manifest gate redirect: if the gate found a compatible sibling,
        # adopt it instead of the auto-invented name. Skip the data generation
        # step entirely — the sibling already has the rep on disk.
        if create_result.get("status") == "blocked_use_existing" and create_result.get("effective_manifest_name"):
            redirected = create_result["effective_manifest_name"]
            log(f"[Innovation Team] Manifest gate redirected '{new_manifest_name}' -> '{redirected}'. "
                f"Skipping data generation (sibling already has '{representation_type}' on disk).")
            return redirected

        # Halt request: tool refused to act and asked the user to intervene.
        if create_result.get("status") == "blocked_user_input_required":
            log(f"[Innovation Team] Manifest creation halted by gate: {create_result.get('message')}")
            return None

        if create_result.get("status") != "completed":
            log(f"[Innovation Team] ERROR: Failed to create manifest '{new_manifest_name}': {create_result.get('message')}")
            return None

        log(f"[Innovation Team] Manifest '{new_manifest_name}' created. Generating {representation_type} representation...")

        # A2: Pre-process data so run_training_trial finds the .npy files
        gen_result = self.execute_tool_func("generate_representation", {
            "manifest_name": new_manifest_name,
            "representation_type": representation_type,
        })
        if gen_result.get("status") == "completed":
            log(f"[Innovation Team] Representation generated successfully for '{new_manifest_name}'.")
        else:
            log(f"[Innovation Team] WARNING: generate_representation failed: {gen_result.get('message')}. Trial may fail.")

        return new_manifest_name


class WebResearchCollaborator:
    """
    An advanced multi-agent team for deep web research.
    1. Chief Researcher: Formulates a list of deep research questions from a simple query.
    2. Analyst: Iteratively searches for each question, compiling a "research dossier".
    3. Writer: Synthesizes the entire dossier into a final, citable report.
    """

    def __init__(self, main_agent: ConversationalAgent, search_tool_function: callable):
        self.main_agent = main_agent
        self.search_tool = search_tool_function
        self.report_dir = PERSISTENT_PATHS["research_reports_dir"]
        os.makedirs(self.report_dir, exist_ok=True)
        log(f"[Web Research Team] Advanced team initialized. Reports will be saved to {self.report_dir}")

    def _run_sub_agent(self, system_prompt: str, conversation_history: list) -> str:
        response, _ = self.main_agent.run_turn(system_prompt, conversation_history)
        return response

    def run_collaboration(self, query: str, max_sites: int) -> dict:
        log("[Web Research Team] Starting deep research collaboration...")

        # --- 1. AGENT 1: CHIEF RESEARCHER (Query Formulation) ---
        log("[Web Research Team] Calling Chief Researcher to formulate a deep research plan...")
        refiner_prompt = f"""
        You are a Chief Research Scientist. Your task is to take a high-level user query and break it down into a list of 5 specific, deep, and diverse sub-queries.
        
        **CRITICAL INSTRUCTION:**
        You MUST include **academic keywords** (e.g., "arxiv", "scholar", "paper", "survey", "state of the art", "SOTA") in at least 2 of your queries to ensure we find rigorous scientific sources.

        Focus on different facets of the topic:
        - Foundational concepts or definitions.
        - Advanced techniques or state-of-the-art (SOTA).
        - Common challenges, pitfalls, or limitations.
        - Practical applications or code examples.
        - Comparisons to alternative methods.

        User's High-Level Query: "{query}"

        Your output MUST be a JSON list of 5 strings.
        Example:
        [
          "query 1...",
          "query 2...",
          "query 3...",
          "query 4...",
          "query 5..."
        ]
        """
        raw_plan_response = self._run_sub_agent(
            "You are a Chief Research Scientist. Output ONLY the JSON list of 5 sub-queries.",
            [{"role": "user", "parts": [refiner_prompt]}],
        )

        try:
            # Clean potential markdown code blocks
            cleaned_response = re.sub(r"^```(json)?\s*|\s*```$", "", raw_plan_response.strip(), flags=re.DOTALL)
            research_plan = json.loads(cleaned_response)
            if not isinstance(research_plan, list) or not all(isinstance(q, str) for q in research_plan):
                raise ValueError("Output is not a list of strings.")
        except Exception as e:
            log(
                f"[Web Research Team] ERROR: Chief Researcher failed to create a valid JSON plan: {e}. Falling back to a simple search."
            )
            research_plan = [query]  # Fallback to the original query

        log(f"[Web Research Team] Research plan formulated with {len(research_plan)} sub-queries.")

        # --- 2. AGENT 2: ANALYST (Recursive Search & Dossier Creation) ---
        log(
            f"[Web Research Team] Calling Analyst to execute {len(research_plan)} searches (max {max_sites} results per search)..."
        )
        research_dossier = ""
        all_sources = []
        source_counter = 1

        try:
            for i, sub_query in enumerate(research_plan):
                log(f'[Web Research Team] Analyst searching for: ({i+1}/{len(research_plan)}) "{sub_query}"')
                # Run the injected search tool
                search_results = self.search_tool(queries=[sub_query], max_results_per_query=max_sites)

                if not search_results:
                    research_dossier += f'--- No results found for sub-query: "{sub_query}" ---\n\n'
                    continue

                research_dossier += f'--- Snippets for sub-query: "{sub_query}" ---\n\n'

                # --- RECURSIVE DEEP DIVE LOGIC ---
                # Identify the top result to potentially "deep dive" into
                if search_results:
                    top_result = search_results[0]
                    title = top_result.get("source_title", "")
                    snippet = top_result.get("snippet", "")
                    
                    # Simple heuristic: if snippet mentions "paper", "arxiv", "method", "algorithm", dive deeper
                    if any(k in snippet.lower() or k in title.lower() for k in ["paper", "arxiv", "method", "algorithm", "novel"]):
                        log(f"[Web Research Team] 🔍 Deep Dive triggered for: '{title}'")
                        deep_dive_query = f"details of {title} method algorithm"
                        deep_results = self.search_tool(queries=[deep_dive_query], max_results_per_query=2)
                        if deep_results:
                            search_results.extend(deep_results) # Add deep dive results to the list
                            research_dossier += f"--- [DEEP DIVE] Extra context for '{title}' ---\n"

                for r in search_results:
                    snippet = r.get("snippet", "No snippet available.")
                    title = r.get("source_title", "No Title")
                    url = r.get("url", "No URL")

                    # Add to dossier for the Writer
                    research_dossier += f"Source [{source_counter}]: {title}\n"
                    research_dossier += f"URL: {url}\n"
                    research_dossier += f"Snippet: {snippet}\n---\n"

                    # Add to sources list for citation
                    all_sources.append({"id": source_counter, "title": title, "url": url})
                    source_counter += 1
                research_dossier += "\n"

            if not all_sources:
                return {
                    "status": "completed",
                    "report": "No web results found for the query or its sub-queries.",
                    "filename": None,
                }

            log(f"[Web Research Team] Analyst compiled a dossier with {len(all_sources)} total snippets.")

        except Exception as e:
            log(f"[Web Research Team] ERROR during iterative search: {e}\n{traceback.format_exc()}")
            return {"status": "error", "message": f"Web research failed during search phase: {e}"}

        # --- 3. AGENT 3: WRITER (Report Generation) ---
        log("[Web Research Team] Calling Writer to synthesize the report...")
        writer_prompt = f"""
        You are a Scientific Writer. Your task is to write a comprehensive, academic-style report based on the provided research dossier.
        
        **User Query:** "{query}"
        
        **Research Dossier:**
        {research_dossier}
        
        **Instructions:**
        1.  **Structure:** Use standard markdown headers (#, ##). Include sections for:
            - **Executive Summary**: High-level overview.
            - **Key Findings**: Detailed analysis of the gathered info.
            - **Novelty & SOTA**: Explicitly highlight any State-of-the-Art methods or novel approaches found.
            - **Code/Implementation Details**: Any practical details found.
            - **References**: List the sources used.
        2.  **Citations:** You MUST cite your sources using the [ID] format (e.g., [1], [3]) whenever you state a fact from the dossier.
        3.  **Tone:** Professional, objective, and detailed.
        """
        
        final_report = self._run_sub_agent(
            "You are a Scientific Writer. Write a detailed, cited report.",
            [{"role": "user", "parts": [writer_prompt]}]
        )
        
        # Save the report to a file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"research_report_{timestamp}.md"
        filepath = os.path.join(self.report_dir, filename)
        
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(final_report)
            log(f"[Web Research Team] Report saved to {filepath}")
            return {
                "status": "completed",
                "report": final_report,
                "filename": filename,
                "filepath": filepath
            }
        except Exception as e:
            log(f"[Web Research Team] ERROR saving report: {e}")
            return {"status": "error", "message": f"Failed to save report: {e}"}




class CreativeHypothesisCollaborator:
    """
    Un team di agenti per generare e selezionare autonomamente nuove idee.
    1. Brainstormer: Genera 5 idee creative (alta temperatura).
    2. Critic: Valuta i pro e i contro di ogni idea in base ai dati attuali.
    3. Decider: Sceglie l'idea migliore da perseguire.
    """

    def __init__(self, main_agent: ConversationalAgent):
        self.main_agent = main_agent

    def _run_sub_agent(self, system_prompt: str, conversation_history: list, temperature: float = 0.7) -> str:
        # Modifica temporanea per passare la temperatura (se l'API la supporta)
        # Nota: l'attuale 'run_turn' non passa la temperatura, quindi questo è un
        # design concettuale. Per ora, useremo prompt diversi.
        response, _ = self.main_agent.run_turn(system_prompt, conversation_history)
        return response

    def run_collaboration(self, initial_query: str, current_data_context: str, strategic_log: List[dict]) -> dict:
        log("[Creative Team] Starting collaboration...")

        # --- 1. AGENT 1: BRAINSTORMER ---
        log("[Creative Team] Calling Brainstormer...")
        brainstormer_prompt = f"""
        You are a highly creative and unconventional research scientist.
        Your task is to generate 5 distinct, creative, and testable hypotheses based on the user's high-level query.
        Think "outside the box." Consider radical changes to architecture, data representation, or training. 
        You can suggest using 'Neural Architecture Search' (NAS) or 'Supernet' to rapidly validate complex architectural ideas.
        Generate 5 responses with their corresponding probabilities, sampled from the full distribution
        
        **Query:**
        {initial_query}
        
        **Output Format:**
        You MUST output a JSON list of 5 strings.
        Example:
        [
          "Hypothesis 1: ...",
          "Hypothesis 2: ...",
          "Hypothesis 3: ...",
          "Hypothesis 4: ...",
          "Hypothesis 5: ..."
        ]
        """
        # Usiamo una temperatura più alta per la creatività
        raw_ideas_response = self._run_sub_agent(
            "You are a creative scientist. Output ONLY the JSON list of 5 hypotheses.",
            [{"role": "user", "parts": [brainstormer_prompt]}],
            # temperature=1.2 # Concettuale
        )

        try:
            ideas_list = json.loads(raw_ideas_response)
            if not isinstance(ideas_list, list) or len(ideas_list) != 5:
                raise ValueError("Response is not a list of 5 strings.")
        except Exception as e:
            log(f"[Creative Team] ERROR: Brainstormer output was not valid JSON. {e}. Fallback: using raw text.")
            ideas_list = [
                line.strip()
                for line in raw_ideas_response.split("\n")
                if line.strip() and (line.strip().startswith('"') or line.strip().startswith("Hypothesis"))
            ]
            if not ideas_list:
                return {"status": "error", "message": "Creative agent failed to generate ideas."}

        # --- 2. AGENT 2: CRITIC ---
        log("[Creative Team] Calling Critic...")
        strategic_log_summary = "Strategic log is empty."
        if strategic_log:
            try:
                # Riassumiamo il log strategico per non sovraccaricare il prompt
                df = pd.DataFrame(strategic_log)
                tool_effectiveness = (
                    df.groupby("tool_name")["improvement"].mean().sort_values(ascending=False).to_dict()
                )
                strategic_log_summary = f"Average improvement per tool: {tool_effectiveness}"
            except Exception:
                pass  # Usa il default

        critic_prompt = f"""
        You are a pragmatic and data-driven project lead.
        Your task is to analyze the 5 creative hypotheses provided and write a "Pro vs. Con" analysis for each one.
        Base your analysis *only* on the provided experiment data and strategic log summary.
        
        **Current Experiment Data:**
        {current_data_context}
        
        **Strategic Log Summary:**
        {strategic_log_summary}

        **Hypotheses to Evaluate:**
        """
        for i, idea in enumerate(ideas_list):
            critic_prompt += f"{i+1}. {idea}\n"

        critic_prompt += "\n**Your Analysis:**\n"

        evaluations = self._run_sub_agent(
            "You are a pragmatic critic. Provide a Pro/Con analysis for each hypothesis.",
            [{"role": "user", "parts": [critic_prompt]}],
        )
        log("[Creative Team] Critic has completed evaluations.")

        # --- 3. AGENT 3: AUTONOMOUS DECIDER ---
        log("[Creative Team] Calling Autonomous Decider...")
        decider_prompt = f"""
        You are the final Decider for the research team.
        You have received 5 hypotheses and a critical analysis from your project lead.
        Your task is to:
        1.  Review the hypotheses and the analysis.
        2.  Select the **single best hypothesis that the cluster can ACTUALLY EXECUTE**.
        3.  Translate it into a concrete, executable action with a clear directive.

        **Hypotheses and Analysis:**
        {evaluations}

        ## EXECUTION REALITY — the cluster can only do these things
        A chosen hypothesis is useless unless it maps to one of these real actions:
        - `propose_architecture_upgrade(base_file, directive)` — MUTATE the current custom architecture (add/remove layers, residual / attention / normalization blocks, change widths or depths). Use this for ANY structural change.
        - `propose_intelligent_architecture` — build a brand-new architecture from scratch.
        - `run_training_trial` / `run_strategic_optuna_sweep` — tune REAL hyperparameters: batch_norm, lr, dropout, num_conv_layers, kernel_size, batch_size, optimizer, epochs.
        - `run_training_trial_with_finetune` — domain adaptation; real knobs: ft_unfreeze_last_n, ft_lr, ft_backbone_lr_scale, ft_epochs, ft_reinitialize_head.

        ## FORBIDDEN — no tool implements these (the worker's training loop is fixed)
        Do NOT choose a hypothesis that depends on a custom training objective: contrastive loss, adversarial / domain-adversarial (DANN), optimal transport / Wasserstein, neural ODE, MixUp / CutMix, IRM, domain-conditional BatchNorm, fractal / self-similar schedules. If the best idea needs one of these, pick the closest EXECUTABLE structural variant (via propose_architecture_upgrade) or mark it out_of_scope.

        **Your Output (follow EXACTLY):**
        Chosen Hypothesis: <full text of the selected hypothesis>
        Justification: <why it is the most promising AND executable choice given the data>
        ACTION_TYPE: <one of: architecture_upgrade | hyperparameter | finetune | out_of_scope>
        DIRECTIVE: <one concrete instruction the downstream tool can act on, e.g. "add a squeeze-and-excitation block after the last conv stage" or "sweep ft_unfreeze_last_n in [2,4] with ft_lr=3e-5">
        [via <tool>, <real_param>=<value_or_range>]
        """

        final_choice = self._run_sub_agent(
            "You are the Decider. Select one executable hypothesis, justify it, and emit the ACTION_TYPE + DIRECTIVE tags.",
            [{"role": "user", "parts": [decider_prompt]}],
        )
        log("[Creative Team] Autonomous Decider has made a selection.")

        # Parse the structured action tags so downstream routing can act on the choice
        # (architecture_upgrade -> Innovation Team directly; else -> Theorist feed-forward).
        action_type = "unparsed"
        directive = ""
        try:
            _m = re.search(r"ACTION_TYPE:\s*([A-Za-z_]+)", final_choice)
            if _m:
                action_type = _m.group(1).strip().lower()
            _m2 = re.search(r"DIRECTIVE:\s*(.+)", final_choice)
            if _m2:
                directive = _m2.group(1).strip()
        except Exception:
            pass

        return {
            "status": "completed",
            "chosen_hypothesis_and_justification": final_choice,
            "action_type": action_type,
            "directive": directive,
        }


class ArchitectureComparisonCollaborator:
    """
    Un team di agenti specializzati per comparare due architetture.
    1. Data Gatherer: Esegue i tool analitici per entrambe le architetture.
    2. Semantic Summarizer: Esegue la ricerca semantica per entrambe e riassume.
    3. Synthesizer: Unisce i dati analitici e semantici per rispondere alla query.
    """

    def __init__(self, main_agent: ConversationalAgent, tool_executor_func: callable):
        self.main_agent = main_agent
        self.execute_tool = tool_executor_func  # Funzione da app.py per chiamare i tool

    def _run_sub_agent(self, system_prompt: str, conversation_history: list) -> str:
        response, _ = self.main_agent.run_turn(system_prompt, conversation_history)
        return response

    def run_collaboration(self, arch_A: str, arch_B: str, query: str) -> dict:
        log(f"[Comparison Team] Starting collaboration: {arch_A} vs {arch_B}")

        # --- 1. AGENT 1: DATA GATHERER (eseguito direttamente) ---
        log("[Comparison Team] Gathering analytical data...")
        analytical_report = f"**Analytical Comparison Report: {arch_A} vs {arch_B}**\n\n"

        try:
            # Raccoglie dati per Arch A
            analytical_report += f"--- {arch_A} Analysis ---\n"
            corr_A = self.execute_tool("get_correlation_matrix", {"architecture_filter": arch_A})
            tradeoff_A = self.execute_tool("analyze_hyperparameter_tradeoffs", {"architecture_filter": arch_A})
            analytical_report += f"Correlation: {json.dumps(corr_A.get('correlation', 'N/A'))}\n"
            analytical_report += f"Tradeoffs: {json.dumps(tradeoff_A.get('tradeoffs', 'N/A'))}\n\n"

            # Raccoglie dati per Arch B
            analytical_report += f"--- {arch_B} Analysis ---\n"
            corr_B = self.execute_tool("get_correlation_matrix", {"architecture_filter": arch_B})
            tradeoff_B = self.execute_tool("analyze_hyperparameter_tradeoffs", {"architecture_filter": arch_B})
            analytical_report += f"Correlation: {json.dumps(corr_B.get('correlation', 'N/A'))}\n"
            analytical_report += f"Tradeoffs: {json.dumps(tradeoff_B.get('tradeoffs', 'N/A'))}\n\n"

            log("[Comparison Team] Analytical data gathered.")

        except Exception as e:
            log(f"[Comparison Team] ERROR gathering analytical data: {e}")
            return {"status": "error", "message": f"Failed to gather analytical data: {e}"}

        # --- 2. AGENT 2: SEMANTIC SUMMARIZER ---
        log("[Comparison Team] Gathering semantic memories...")
        semantic_report = f"**Semantic Memory Comparison: {arch_A} vs {arch_B}**\n\n"

        try:
            # Memorie per Arch A
            mem_A = self.execute_tool(
                "recall_relevant_memories", {"query": f"key findings for {arch_A} related to {query}"}
            )
            semantic_report += f"--- {arch_A} Memories ---\n"
            semantic_report += json.dumps(mem_A.get("memories", "No relevant memories found.")) + "\n\n"

            # Memorie per Arch B
            mem_B = self.execute_tool(
                "recall_relevant_memories", {"query": f"key findings for {arch_B} related to {query}"}
            )
            semantic_report += f"--- {arch_B} Memories ---\n"
            semantic_report += json.dumps(mem_B.get("memories", "No relevant memories found.")) + "\n\n"

            log("[Comparison Team] Semantic memories gathered.")

        except Exception as e:
            log(f"[Comparison Team] ERROR gathering semantic data: {e}")
            return {"status": "error", "message": f"Failed to recall memories: {e}"}

        # --- 3. AGENT 3: SYNTHESIZER ---
        log("[Comparison Team] Calling Synthesizer...")
        synthesizer_prompt = f"""
        You are an expert research analyst. Your task is to synthesize the following analytical and semantic data to answer the user's query.
        Provide a concise, data-driven recommendation.

        **User Query:**
        "{query}"

        **1. Analytical Data Report:**
        {analytical_report}
            
        **2. Semantic Memory Report:**
        {semantic_report}
            
        **Your Synthesis:**
        Based on all the data, answer the query. Conclude with a clear hypothesis or recommendation.
        """

        final_synthesis = self._run_sub_agent(
            "You are an expert research analyst. Synthesize the provided data to answer the query.",
            [{"role": "user", "parts": [synthesizer_prompt]}],
        )
        log("[Comparison Team] Synthesis complete.")

        return {"status": "completed", "comparison_report": final_synthesis}



class StrategicGraphEvaluatorCollaborator:
    """
    A specialized team for evaluating the StrategicGraphMemory.
    Implements a 'Team of 2' approach:
    1. The Graph Mathematician: Quantitative analysis of the graph structure and metrics.
    2. The Strategic Narrator: Qualitative interpretation and high-level strategy.
    """

    def __init__(self, main_agent: ConversationalAgent, execute_tool_func: Optional[Callable] = None):
        self.main_agent = main_agent
        self.execute_tool_func = execute_tool_func

    def _run_sub_agent(self, system_prompt: str, conversation_history: list) -> str:
        response, _ = self.main_agent.run_turn(system_prompt, conversation_history)
        return response

    async def run_collaboration(self, thread_safe_state: Optional[Dict[str, Any]] = None, strategic_intent: Optional[str] = None) -> dict:
        log("[Strategic Graph Evaluator Team] Starting collaboration...")

        # --- 0. GET GRAPH DATA ---
        graph_data = {"nodes": [], "edges": []}
        strategic_graph_obj = None
        
        # Try to get the actual object from session state or thread_safe_state
        try:
             import sys
             if "streamlit" in sys.modules:
                 st = sys.modules["streamlit"]
                 if hasattr(st, "session_state") and hasattr(st.session_state, "strategic_graph_memory"):
                     strategic_graph_obj = st.session_state.strategic_graph_memory
                     graph_data = strategic_graph_obj.get_strategic_graph()
        except Exception:
             pass

        if not graph_data["nodes"] and self.execute_tool_func:
             try:
                 graph_data = self.execute_tool_func("get_strategic_graph", {})
             except Exception as e:
                 log(f"[Strategic Graph Evaluator] Failed to get graph data via tool: {e}")

        graph_json = json.dumps(graph_data)
        
        # --- 1. THE GRAPH MATHEMATICIAN (LLM) ---
        log("[Strategic Graph Evaluator] Calling The Graph Mathematician...")
        math_prompt = f"""
        You are The Graph Mathematician. Your goal is to analyze the raw data of the Strategic Graph and output a purely quantitative report.
        
        **Graph Data:**
        {graph_json}
        
        **Analysis Required:**
        1.  **Global Metrics:** Total nodes, total edges, average reward, graph density.
        2.  **Node Centrality:** Which nodes (states) are most visited? Which have the highest value?
        3.  **Edge Analysis:** Which actions (edges) yield the highest rewards? Which are "dead ends"?
        4.  **Exploration Metrics:** What percentage of the graph is "high confidence" vs. "unexplored"?
        
        **Output Format:**
        Provide a structured list of metrics and mathematical observations. Do NOT give strategic advice yet.
        """
        
        math_report = self._run_sub_agent(
            "You are an expert Graph Mathematician. Analyze the data quantitatively.",
            [{"role": "user", "parts": [math_prompt]}]
        )
        
        # --- 2. THE MOSAN ADVISOR (Algorithmic / Pareto) ---
        algo_advice = ""
        if strategic_graph_obj and thread_safe_state:
            try:
                from context_planner import StrategicAdvisorTeam
                log("[Strategic Evaluator] Calling MOSAN Advisor...")
                # Instantiate Advisor
                advisor = StrategicAdvisorTeam(self.main_agent, strategic_graph_obj)
                
                # Create a minimal context summary if not present
                context_summary = {"task_type": "optimization"} 
                
                # Generate Algorithmic Advice
                algo_advice = await advisor.generate_advice(
                    context_summary, 
                    thread_safe_state, 
                    strategic_intent
                )
                log("[Strategic Evaluator] MOSAN Advisor advice generated.")
                
            except Exception as e:
                 log(f"[Strategic Evaluator] MOSAN Advisor failed: {e}")
                 algo_advice = f"(Algorithmic Advisor unavailable: {e})"

        # --- 3. THE STRATEGIC NARRATOR (LLM) ---
        log("[Strategic Graph Evaluator] Calling The Strategic Narrator...")
        narrator_prompt = f"""
        You are The Strategic Narrator. Your goal is to translate the Mathematician's quantitative report into a high-level strategic narrative.
        
        **Strategic Intent:** {strategic_intent if strategic_intent else "BALANCED / OPTIMIZATION"}
        
        **Mathematician's Report:**
        {math_report}
        
        **Your Task:**
        1.  **Interpret the Numbers:** What do the high-value nodes represent?
        2.  **Identify Bottlenecks:** Are we stuck?
        3.  **Propose Strategy:** Based on the graph structure AND the intent '{strategic_intent}', where should we explore next?
        
        **Output Format:**
        A concise, natural language summary providing clear strategic direction.
        """
        
        strategic_narrative = self._run_sub_agent(
            "You are a Strategic Narrator. Translate metrics into strategy.",
            [{"role": "user", "parts": [narrator_prompt]}]
        )
        
        final_report = f"{strategic_narrative}\n\n**MOSAN Strategic Advice:**\n{algo_advice}\n\n**Quantitative Details:**\n{math_report}"
        
        log("[Strategic Graph Evaluator] Collaboration complete.")
        return {
            "status": "completed",
            "report": final_report,
            "math_details": math_report
        }


class ScientificReportGeneratorCollaborator:
    """
    A team of agents to generate a scientific report from the experiment data.
    """

    def __init__(self, main_agent: ConversationalAgent, tool_executor_func: callable, chunk_token_limit: int = 50000):
        self.main_agent = main_agent
        self.execute_tool = tool_executor_func
        self.chunk_token_limit = chunk_token_limit

    def _flatten_logs(self, logs: list) -> list:
        """Recursively flattens a nested list of logs."""
        flat_logs = []
        for item in logs:
            if isinstance(item, list):
                flat_logs.extend(self._flatten_logs(item))
            else:
                flat_logs.append(item)
        return flat_logs

    def _estimate_tokens(self, text: str) -> int:
        return len(text) // 4

    def _chunk_history(self, logs: list, max_tokens: int) -> list:
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for msg in logs:
            msg_str = json.dumps(msg)
            msg_tokens = self._estimate_tokens(msg_str)
            
            if current_tokens + msg_tokens > max_tokens and current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_tokens = 0
            
            current_chunk.append(msg)
            current_tokens += msg_tokens
            
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks

    async def _summarize_chunk(self, chunk: list, index: int, total: int) -> str:
        chunk_text = json.dumps(chunk, indent=2)
        prompt = f"""
        You are a Research Assistant. Summarize this segment (Part {index+1}/{total}) of the research log.
        Focus on:
        - Key actions taken.
        - Important findings or errors.
        - Strategic decisions made.
        Be concise.
        """
        log(f"[Scientific Report Generator] Summarizing chunk {index+1}/{total} ({self._estimate_tokens(chunk_text)} tokens)...")
        summary, _ = await asyncio.to_thread(self.main_agent.run_turn, prompt, [{"role": "user", "parts": [f"Log Segment:\n{chunk_text}"]}])
        return summary

    def _stage_chunks(self, logs: list) -> int:
        # 1. Setup Staging Directory
        staging_dir = os.path.join("processed_data", "temp_chunks")
        # Ensure absolute path for clarity in logs
        staging_dir = os.path.abspath(staging_dir)
        
        if os.path.exists(staging_dir):
            import shutil
            try:
                shutil.rmtree(staging_dir)
            except Exception as e:
                log(f"[Scientific Report Generator] WARNING: Failed to clean staging dir: {e}")
        os.makedirs(staging_dir, exist_ok=True)

        log(f"[Scientific Report Generator] Staging Directory: {staging_dir}")
        log(f"[Scientific Report Generator] Input Logs: {len(logs)} messages.")
        
        # Debug: Check first item type
        if logs:
            log(f"[Scientific Report Generator] First message type: {type(logs[0])}")

        log(f"[Scientific Report Generator] Splitting messages into chunks (limit: {self.chunk_token_limit} tokens)...")
        chunks = self._chunk_history(logs, self.chunk_token_limit)
        
        if not chunks:
            log("[Scientific Report Generator] No chunks created.")
            return 0

        # 2. Save Chunks to Disk
        for i, chunk in enumerate(chunks):
            chunk_filename = os.path.join(staging_dir, f"chunk_{i}.json")
            try:
                with open(chunk_filename, "w") as f:
                    json.dump(chunk, f, indent=2)
                log(f"[Scientific Report Generator] Saved {chunk_filename} ({len(chunk)} msgs)")
            except Exception as e:
                log(f"[Scientific Report Generator] ERROR saving {chunk_filename}: {e}")
        
        log(f"[Scientific Report Generator] Staged {len(chunks)} chunks to {staging_dir}.")
        return len(chunks)

    async def _process_staged_chunks(self, num_chunks: int) -> None:
        staging_dir = os.path.join("processed_data", "temp_chunks")
        
        for i in range(num_chunks):
            chunk_file = os.path.join(staging_dir, f"chunk_{i}.json")
            summary_file = os.path.join(staging_dir, f"summary_{i}.txt")
            
            if os.path.exists(summary_file):
                log(f"[Scientific Report Generator] Chunk {i+1}/{num_chunks} already summarized. Skipping.")
                continue

            log(f"[Scientific Report Generator] Processing chunk {i+1}/{num_chunks} from {chunk_file}...")
            
            try:
                # Read chunk
                with open(chunk_file, "r") as f:
                    chunk_data = json.load(f)
                
                # Summarize
                summary = await self._summarize_chunk(chunk_data, i, num_chunks)
                
                # Save Summary
                with open(summary_file, "w") as f:
                    f.write(summary)
                
                log(f"[Scientific Report Generator] Chunk {i+1} summary saved.")
            except Exception as e:
                log(f"[Scientific Report Generator] ERROR processing chunk {i+1}: {e}")
                # Continue to next chunk even if this one fails

    def _aggregate_summaries(self, num_chunks: int) -> str:
        staging_dir = os.path.join("processed_data", "temp_chunks")
        summaries = []
        
        for i in range(num_chunks):
            summary_file = os.path.join(staging_dir, f"summary_{i}.txt")
            if os.path.exists(summary_file):
                with open(summary_file, "r") as f:
                    summaries.append(f.read())
            else:
                summaries.append(f"[MISSING SUMMARY FOR CHUNK {i+1}]")
        
        combined_summary = "\n\n".join([f"--- Part {i+1}/{num_chunks} ---\n{s}" for i, s in enumerate(summaries)])
        
        # Save aggregated summary
        aggregated_file = os.path.join(staging_dir, "aggregated_summary.txt")
        with open(aggregated_file, "w") as f:
            f.write(combined_summary)
        
        log(f"[Scientific Report Generator] Aggregated summary saved to {aggregated_file}.")
        return combined_summary

    async def run_sequential_summarization(self, logs: list) -> str:
        if not isinstance(logs, list):
            return str(logs)[:10000] # Fallback for string logs

        # Flatten logs to ensure we are chunking messages, not sessions
        log(f"[Scientific Report Generator] Flattening logs...")
        flat_logs = self._flatten_logs(logs)
        log(f"[Scientific Report Generator] Flattened {len(logs)} items into {len(flat_logs)} messages.")

        # Step 1: Stage
        num_chunks = self._stage_chunks(flat_logs)
        if num_chunks == 0:
            return "No conversation history available."

        # Step 2: Process
        await self._process_staged_chunks(num_chunks)

        # Step 3: Aggregate
        combined_summary = self._aggregate_summaries(num_chunks)
        return combined_summary

    def _run_sub_agent(self, system_prompt: str, conversation_history: list) -> str:
        response, _ = self.main_agent.run_turn(system_prompt, conversation_history)
        return response

    async def run_data_gatherer(self, dataset: str, architecture: str, logs: list, memory: Any, results_log: list, strategic_log: list) -> str:
        # Gathers and formats data for the analyst
        log("[Scientific Report Generator Team] Gathering data...")
        
        # Process logs (conversation history) with Sequential Summarization
        formatted_logs = await self.run_sequential_summarization(logs)

        # Format results log
        formatted_results = json.dumps(results_log[-20:], indent=2) if isinstance(results_log, list) else str(results_log)

        gathered_data = {
            "dataset": dataset,
            "architecture": architecture,
            "recent_conversation": formatted_logs,
            "recent_trials": formatted_results,
            "strategic_log": str(strategic_log)[:2000],
            "memory_snapshot": str(memory)[:2000] # Placeholder for vector memory dump
        }
        return f"Data Context:\n{json.dumps(gathered_data, indent=2)}"

    async def run_data_analyser(self, data: str) -> str:
        analyser_system_prompt = """
        You are a Data Analyst. Your task is to analyze the provided data and extract key insights.
        - Identify trends and patterns.
        - Find correlations and causations.
        - Summarize the most important findings.
        """
        log("[Scientific Report Generator Team] Calling Data Analyser...")
        analysis, _ = self.main_agent.run_turn(
            analyser_system_prompt, [{"role": "user", "parts": [f"Here is the data:\n{data}"]}]
        )
        return analysis

    async def run_report_writer(self, analysis: str) -> str:
        writer_system_prompt = """
        You are a Scientific Writer. Your task is to write a scientific report based on the provided analysis.
        The report should have the following sections:
        - Abstract
        - Introduction
        - Methods
        - Results
        - Conclusion
        """
        log("[Scientific Report Generator Team] Calling Report Writer...")
        report, _ = self.main_agent.run_turn(
            writer_system_prompt, [{"role": "user", "parts": [f"Here is the analysis:\n{analysis}"]}]
        )
        return report

    async def run_collaboration(self, dataset: str, architecture: str, logs: list, memory: Any, results_log: list, strategic_log: list) -> dict:
        log("[Scientific Report Generator Team] Starting collaboration...")

        # 1. Gather data
        gathered_data = await self.run_data_gatherer(dataset, architecture, logs, memory, results_log, strategic_log)

        # 2. Analyze data
        analysis = await self.run_data_analyser(gathered_data)

        # 3. Write report
        final_report = await self.run_report_writer(analysis)

        # Save report to file
        report_filename = f"report_{architecture}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        report_path = os.path.join(PERSISTENT_PATHS["research_reports_dir"], report_filename)
        os.makedirs(PERSISTENT_PATHS["research_reports_dir"], exist_ok=True)
        with open(report_path, "w") as f:
            f.write(final_report)
        
        log(f"[Scientific Report Generator Team] Report saved to {report_path}")

        log("[Scientific Report Generator Team] Collaboration complete.")
        return {"status": "completed", "report": final_report, "report_path": report_path}


class FinetuningDatasetGeneratorCollaborator:
    """
    A team of agents to generate a finetuning dataset from the experiment data.
    """

    def __init__(self, main_agent: ConversationalAgent, tool_executor_func: callable):
        self.main_agent = main_agent
        self.execute_tool = tool_executor_func

    def _run_sub_agent(self, system_prompt: str, conversation_history: list) -> str:
        response, _ = self.main_agent.run_turn(system_prompt, conversation_history)
        return response

    async def run_data_gatherer(self, logs: list, memory: Any, results_log: list, strategic_log: list) -> str:
        # Gathers and formats data
        log("[Finetuning Dataset Generator Team] Gathering data...")
        
        formatted_logs = json.dumps(logs, indent=2) if isinstance(logs, list) else str(logs)
        
        gathered_data = {
            "logs": formatted_logs[:10000], # Limit size
            "results_log": str(results_log)[:5000],
            "strategic_log": str(strategic_log)[:2000]
        }
        return f"Data Context:\n{json.dumps(gathered_data, indent=2)}"

    async def run_dataset_formatter(self, data: str) -> str:
        formatter_system_prompt = """
        You are a Data Formatter. Your task is to format the provided data into prompt-completion pairs for LLM finetuning.
        - Each pair should have a "prompt" and a "completion" key.
        - The prompt should be a question or a statement that elicits the completion.
        - The completion should be the desired output from the LLM.
        - The output should be a list of JSON objects, each representing a prompt-completion pair.
        """
        log("[Finetuning Dataset Generator Team] Calling Dataset Formatter...")
        formatted_data, _ = self.main_agent.run_turn(
            formatter_system_prompt, [{"role": "user", "parts": [f"Here is the data:\n{data}"]}]
        )
        return formatted_data

    async def run_dataset_writer(self, data: str) -> str:
        # In a real implementation, this would write the data to a file in the desired format (e.g., JSONL)
        # For this example, we'll just return the data as a string
        return data

    async def run_collaboration(self, logs: list, memory: Any, results_log: list, strategic_log: list) -> dict:
        log("[Finetuning Dataset Generator Team] Starting collaboration...")

        # 1. Gather data
        gathered_data = await self.run_data_gatherer(logs, memory, results_log, strategic_log)

        # 2. Format data
        formatted_data = await self.run_dataset_formatter(gathered_data)

        # 3. Write dataset
        final_dataset = await self.run_dataset_writer(formatted_data)

        log("[Finetuning Dataset Generator Team] Collaboration complete.")
        return {"status": "completed", "dataset": final_dataset}


class AnalystCollaborator:
    def __init__(self, main_agent: ConversationalAgent):
        self.main_agent = main_agent

    def run_analysis(self, context: str) -> str:
        prompt = f"""
        You are the Lead Data Analyst. Analyze the following data context objectively.
        - Identify trends, outliers, and key metrics.
        - Be concise and factual.
        
        Context:
        {context}
        """
        response, _ = self.main_agent.run_turn(prompt, [])
        return response


class StrategistCollaborator:
    def __init__(self, main_agent: ConversationalAgent):
        self.main_agent = main_agent

    def formulate_strategy(self, analysis: str, history_summary: str) -> str:
        prompt = f"""
        You are the Lead Strategist. Based on the analysis and history, formulate a high-level strategic goal.
        - Focus on the "Why" and "What", not the specific tools (let the Decider handle "How").
        - Propose a clear direction (e.g., "Pivot to architecture X", "Deepen hyperparameter tuning").
        
        Analysis:
        {analysis}
        
        History Summary:
        {history_summary}
        """
        response, _ = self.main_agent.run_turn(prompt, [])
        return response


class SummarizerCollaborator:
    def __init__(self, main_agent: ConversationalAgent):
        self.main_agent = main_agent

    def summarize_session(self, history: List[Dict[str, Any]]) -> str:
        history_text = "\n".join([f"{msg['role']}: {msg.get('content','')}" for msg in history[-20:]]) # Summarize last 20 turns
        prompt = f"""
        Summarize the key findings, decisions, and current status from this conversation history.
        Focus on what was learned and what the next steps should be.
        
        History:
        {history_text}
        """
        response, _ = self.main_agent.run_turn(prompt, [])
        return response

# ============================================================================
# NEW HYBRID GRAPH ADVISOR COLLABORATOR
# ============================================================================

class HybridGraphAdvisorCollaborator:
    """
    Orchestrates the Hybrid Graph Intelligence System.
    Combines algorithmic pathfinding (A*) with expert LLM analysis.
    """
    
    def __init__(self, main_agent: ConversationalAgent, execute_tool_func: Callable):
        self.main_agent = main_agent
        self.execute_tool_func = execute_tool_func
        
    async def run_collaboration(
        self, 
        thread_safe_state: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Runs the full hybrid graph analysis pipeline.
        
        Phase 1: Algorithmic Intelligence (Pathfinding)
        Phase 2: Expert Intelligence (Local Neighborhood Analysis)
        Phase 3: Synthesis (Strategic Advice Generation)
        """
        from tools import get_strategic_path, get_local_transitions
        
        log("[HybridAdvisor] Starting hybrid graph analysis...")
        
        # --- PHASE 1: Algorithmic Intelligence ---
        # Get optimal path using A* on Q-values
        path_result = get_strategic_path(thread_safe_state=thread_safe_state)
        
        path_str = "No viable path found."
        if path_result.get("status") == "completed" and path_result.get("path"):
            path = path_result["path"]
            improvement = path_result.get("expected_improvement", 0)
            
            # Format path for LLM
            steps_str = "\n".join([
                f"{step['step']}. {step['action']} (Gain: +{step['expected_gain']}, Conf: {step['confidence']})" 
                for step in path
            ])
            path_str = f"Recommended Path (Effectiveness: +{improvement:.1%}):\n{steps_str}"
            
        # --- PHASE 2: Expert Intelligence Data Gathering ---
        # Get local neighborhood for context
        transitions_result = get_local_transitions(thread_safe_state=thread_safe_state, top_k=5)
        
        transitions_str = "No local transitions available."
        if transitions_result.get("status") == "completed" and transitions_result.get("transitions"):
            trans = transitions_result["transitions"]
            items_str = "\n".join([
                f"{t['rank']}. {t['action']} (Q: {t['q_value']}, Visits: {t['visits']})" 
                for t in trans
            ])
            transitions_str = f"Top Local Options (Ranked by Q-Value):\n{items_str}"
            
        # --- PHASE 3: Expert Analysis & Synthesis ---
        
        # Construct prompt for the Graph Strategist
        system_prompt = """You are the **Graph Strategist**, an expert in navigating the search space of neural network optimization.
Your goal is to synthesize algorithmic pathfinding data with qualitative strategic analysis to provide a single, actionable recommendation.

You have two sources of intelligence:
1. **Algorithmic Path**: A mathematically optimal path based on historical Q-values (exploitation).
2. **Local Options**: The immediate neighborhood of the current state (exploration/exploitation mix).

**Your Output Format must be exactly:**
# Strategic Directive
[One sentence clear instruction]

# Rationale
[Why this is the best move, citing evidence from path or local options]

# Risk Assessment
[Potential downsides or what to watch out for]

# Contingency
[What to do if this fails]
"""

        user_context = f"""
**Current Situation Analysis**
- Algorithmic Path:
{path_str}

- Local Neighborhood Options:
{transitions_str}

**Task:** Compare the algorithmic recommendation with local options. 
- If they align, validate the strategy with high confidence.
- If they disagree, analyze why effectively.
- If both are empty, suggest a broad exploration strategy (e.g., random search or supernet).

Provide your Strategic Directive.
"""
        
        # Call LLM
        response, _ = self.main_agent.run_turn(
            system_prompt=system_prompt,
            conversation_history=[{"role": "user", "content": user_context}],
            model_override=None # Use default model (or consider a cheaper one)
        )
        
        log(f"[HybridAdvisor] Analysis complete. Generated advice.")
        
        return {
            "status": "completed",
            "report": response,
            "algorithmic_path": path_result,
            "expert_analysis": response
        }
