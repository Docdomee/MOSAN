import sys
import os

# Setup path per importare moduli core
# Ensure the project root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from cluster.core.human_interface import HumanInterface
except ImportError:
    # Fallback if cluster package is not directly importable (e.g. during tests)
    from core.human_interface import HumanInterface

def execute(tool_name, python_code, description, reason="Errore persistente"):
    """
    Tool per PROPORRE la creazione di un nuovo tool personalizzato.
    NON esegue il codice. Invia una mail e mette in pausa l'agente.
    """
    interface = HumanInterface()
    return interface.propose_tool(tool_name, python_code, description, reason)
