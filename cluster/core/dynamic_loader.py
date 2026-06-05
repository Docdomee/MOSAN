import os
import importlib.util

class DynamicToolLoader:
    def __init__(self, tools_dir=None):
        if tools_dir is None:
             # Default to cluster/dynamic_tools relative to project root
             # Assuming this script is running from project root or similar
             self.tools_dir = os.path.join("cluster", "dynamic_tools")
        else:
            self.tools_dir = tools_dir

    def load_tools(self):
        loaded_tools = {}

        # 1. Carica il wrapper di creazione (Base Tool)
        try:
            # We try to import it as a module first if it's in the path
            from cluster.dynamic_tools import tool_creator_wrapper
            loaded_tools['create_new_tool'] = tool_creator_wrapper.execute
        except ImportError:
             # Fallback: manual load if not in python path
             wrapper_path = os.path.join(self.tools_dir, "tool_creator_wrapper.py")
             if os.path.exists(wrapper_path):
                 try:
                    spec = importlib.util.spec_from_file_location("tool_creator_wrapper", wrapper_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    loaded_tools['create_new_tool'] = module.execute
                 except Exception as e:
                     print(f"[Loader] Failed manual load of wrapper: {e}")

        # 2. Carica i tool dinamici approvati
        if os.path.exists(self.tools_dir):
            for filename in os.listdir(self.tools_dir):
                if filename.endswith(".py") and filename != "tool_creator_wrapper.py" and not filename.startswith("__"):
                    tool_name = filename[:-3]
                    file_path = os.path.join(self.tools_dir, filename)
                    try:
                        spec = importlib.util.spec_from_file_location(tool_name, file_path)
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        if hasattr(module, 'execute'):
                            loaded_tools[tool_name] = module.execute
                            print(f"[Loader] Tool dinamico caricato: {tool_name}")
                    except Exception as e:
                        print(f"[Loader] Errore caricamento {tool_name}: {e}")

        return loaded_tools
