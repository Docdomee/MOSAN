# Tool argument generation for Phase 2
# Added methods to GraphPlannerAgent for context-aware arg generation

import asyncio
import json
import re
from typing import Dict, Any, List, Optional

# This would be inserted into GraphPlannerAgent class in context_planner.py

class ToolArgumentGenerator:
    """
    Helper class for generating context-aware tool arguments.
    Integrates with GraphPlannerAgent to provide optimal args for each tool in a sequence.
    """
    
    def __init__(self, main_agent: 'ConversationalAgent'):
        self.main_agent = main_agent
        # Tool signatures for validation
        self.tool_schemas = self._load_tool_schemas()
    
    async def generate_args_for_sequence(
        self,
        sequence: List[str],
        context_summary: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate optimal arguments for each tool in a sequence.
        
        Args:
            sequence: List of tool names
            context_summary: Context from ContextCriticAgent
            
        Returns:
            List of arg dicts for each tool (same order as sequence)
        """
        args_list = []
        
        for i, tool_name in enumerate(sequence):
            # Get tool schema if available
            schema = self.tool_schemas.get(tool_name, {})
            
            # Generate args via LLM
            tool_args = await self._generate_single_tool_args(
                tool_name,
                schema,
                context_summary,
                position_in_sequence=i,
                total_steps=len(sequence)
            )
            
            # Validate and fallback if needed
            validated_args = self._validate_args(tool_name, tool_args, schema)
            args_list.append(validated_args)
        
        return args_list
    
    async def _generate_single_tool_args(
        self,
        tool_name: str,
        schema: Dict[str, Any],
        context_summary: Dict[str, Any],
        position_in_sequence: int,
        total_steps: int
    ) -> Dict[str, Any]:
        """Generate args for a single tool using LLM."""
        
        # Build prompt with schema and context
        arg_prompt = f"""You are the **Tool Arguments Generator**. Generate optimal arguments for this tool.

**TOOL**: {tool_name}
**POSITION**: Step {position_in_sequence + 1}/{total_steps} in sequence

**TOOL SCHEMA** (parameters with types and defaults):
{json.dumps(schema, indent=2)}

**CURRENT CONTEXT**:
- Architecture: {context_summary.get('context_prior', 'unknown')}
- Task Type: {context_summary.get('task_type', 'unknown')}
- Active Constraints: {context_summary.get('active_constraints', [])}
- Recent Errors to Avoid: {context_summary.get('errors_to_avoid', [])}

**RULES**:
1. Use schema types (int, float, str, bool, dict, list)
2. Provide sensible defaults if schema missing
3. Align with current task_type and architecture
4. Avoid args that caused errors in errors_to_avoid

**OUTPUT** (JSON only):
{{
  "arg_name_1": value1,
  "arg_name_2": value2,
  ...
}}

For {tool_name}, common args might include:
- manifest_name: str (experiment identifier)
- dataset_id: str (e.g., "data_paper_mine_csv")
- num_conv_layers, learning_rate, batch_size, etc. for training

Output ONLY valid JSON with args, no explanation."""

        try:
            from ui_logger import log
            response, _ = await asyncio.to_thread(
                self.main_agent.run_turn,
                f"You are an args generator for tool {tool_name}. Output valid JSON only.",
                [{"role": "user", "parts": [arg_prompt]}]
            )
            
            # Parse JSON
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                args = json.loads(json_match.group(0))
                return args
            else:
                log(f"[Tool Args] No JSON found in response for {tool_name}, using defaults")
                return {}
                
        except Exception as e:
            from ui_logger import log
            log(f"[Tool Args] Error generating args for {tool_name}: {e}")
            return {}
    
    def _validate_args(
        self,
        tool_name: str,
        args: Dict[str, Any],
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate generated args against schema.
        Fallback to defaults on type mismatches or missing required args.
        """
        validated = {}
        
        if not schema:
            # No schema available, accept all args (risky but flexible)
            return args
        
        # Check each arg in schema
        for arg_name, arg_spec in schema.items():
            required = arg_spec.get("required", False)
            arg_type = arg_spec.get("type", "any")
            default = arg_spec.get("default", None)
            
            # Check if arg provided
            if arg_name in args:
                value = args[arg_name]
                
                # Type validation
                if self._check_type(value, arg_type):
                    validated[arg_name] = value
                else:
                    from ui_logger import log
                    log(f"[Tool Args] Type mismatch for {tool_name}.{arg_name}: expected {arg_type}, got {type(value).__name__}. Using default: {default}")
                    if default is not None:
                        validated[arg_name] = default
            else:
                # Arg not provided
                if required and default is not None:
                    validated[arg_name] = default
                elif default is not None:
                    validated[arg_name] = default
                # If required but no default, omit (tool will error, but that's OK for debugging)
        
        # Also include any extra args not in schema (permissive)
        for arg_name, value in args.items():
            if arg_name not in validated:
                validated[arg_name] = value
        
        return validated
    
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type."""
        type_map = {
            "int": int,
            "float": (int, float),  # Accept int for float
            "str": str,
            "bool": bool,
            "dict": dict,
            "list": list,
            "any": object
        }
        
        expected = type_map.get(expected_type, object)
        return isinstance(value, expected)
    
    def _load_tool_schemas(self) -> Dict[str, Dict]:
        """
        Load tool schemas from tools.py or a schema file.
        
        For now, returns hardcoded schemas for common tools.
        TODO: Auto-extract from tools.py using introspection.
        """
        # Hardcoded schemas for critical tools
        schemas = {
            "run_training_trial": {
                "manifest_name": {"type": "str", "required": False, "default": "trial_auto"},
                "custom_architecture_file": {"type": "str", "required": False, "default": None},
                "representation_type": {"type": "str", "required": False, "default": None},
                "experiment_name": {"type": "str", "required": False, "default": None},
                "num_conv_layers": {"type": "int", "required": False, "default": 4},
                "learning_rate": {"type": "float", "required": False, "default": 0.001},
                "batch_size": {"type": "int", "required": False, "default": 32},
                "max_epochs": {"type": "int", "required": False, "default": 50}
            },
            "run_strategic_optuna_sweep": {
                "n_trials": {"type": "int", "required": True, "default": 20},
                "hyperparameters": {"type": "dict", "required": True, "default": {
                    "learning_rate": {"type": "loguniform", "min": 1e-5, "max": 1e-2},
                    "dropout_rate": {"type": "float", "min": 0.1, "max": 0.6}
                }},
                "experiment_name": {"type": "str", "required": True, "default": "optuna_exp"},
                "optimization_focus": {"type": "str", "required": True, "default": "accuracy"},
                "manifest_name": {"type": "str", "required": True, "default": "data_paper_mine_csv"},
                "custom_architecture_file": {"type": "str", "required": False, "default": None},
                "representation_type": {"type": "str", "required": False, "default": None}
            },
            "run_supernet_search": {
                "dataset_id": {"type": "str", "required": False, "default": "data_paper_mine_csv"},
                "cell_complexity": {"type": "str", "required": False, "default": "medium"},
                "simplified": {"type": "bool", "required": False, "default": False}
            },
            "propose_intelligent_architecture": {
                "architecture_filter": {"type": "str", "required": False, "default": None},
                "inspiration_sources": {"type": "list", "required": False, "default": []}
            },
            "write_architecture_file": {
                "filename": {"type": "str", "required": True, "default": "new_model.py"},
                "code": {"type": "str", "required": True, "default": "# code here"}
            },
            # Add more as needed
        }
        
        return schemas
