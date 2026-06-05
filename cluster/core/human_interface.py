import os
import json
import shutil
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Imports
from cluster.core.google_client import GoogleClient

STAGING_DIR = os.path.join("cluster", "pending_tools")
ACTIVE_DIR = os.path.join("cluster", "dynamic_tools")

class HumanInterface:
    def __init__(self):
        os.makedirs(STAGING_DIR, exist_ok=True)
        os.makedirs(ACTIVE_DIR, exist_ok=True)
        self.google_client = GoogleClient()
        self.recipient_email = os.getenv("EMAIL_RECIPIENT", "admin@localhost")

    def propose_tool(self, tool_name, python_code, description, reason):
        """
        1. Save logic.
        2. Notify (Gmail).
        3. BLOCK and WAIT for approval (Polling).
        4. Return Success message with Feedback.
        """
        safe_name = "".join(x for x in tool_name if x.isalnum() or x == "_").lower()
        file_path = os.path.join(STAGING_DIR, f"{safe_name}.py")
        meta_path = os.path.join(STAGING_DIR, f"{safe_name}.json")

        # Write Code
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(python_code)

        # Write Metadata
        metadata = {
            "name": safe_name,
            "description": description,
            "reason_for_creation": reason,
            "status": "pending_approval"
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # Notify
        self._notify_user(safe_name, description, reason)

        # --- POLLING LOOP (SLURM COMPATIBILITY) ---
        print(f"\n[HITL] ⏳ WAITING FOR APPROVAL via shared filesystem OR Gmail...")
        print(f"Option 1 (Filesystem): python cluster/scripts/approve_tool.py {safe_name}")
        print(f"Option 2 (Gmail): Reply with subject 'APPROVE {safe_name}'")
        
        feedback_msg = self._wait_for_approval(safe_name)
        
        return {
            "status": "completed", # Agent continues!
            "tool_name": safe_name,
            "message": f"Tool '{safe_name}' APPROVED. User Feedback: {feedback_msg}"
        }

    def _wait_for_approval(self, safe_name):
        """
        Polls the filesystem AND Gmail every 30 seconds.
        """
        target_path = os.path.join(ACTIVE_DIR, f"{safe_name}.py")
        feedback_path = os.path.join("cluster", "approval_feedback.json")
        
        while True:
            # 1. Check Filesystem (Manual Script)
            if os.path.exists(target_path):
                print(f"\n[HITL] ✅ Approval Detected (Filesystem)! Resuming...")
                msg = "Approved."
                if os.path.exists(feedback_path):
                    try:
                        with open(feedback_path, "r") as f:
                            data = json.load(f)
                            if data.get("tool_name") == safe_name:
                                msg = data.get("message", "Approved.")
                        os.remove(feedback_path)
                    except Exception: pass
                return msg
            
            # 2. Check GMAIL (Remote Approval)
            feedback = self.google_client.check_for_approval_email(safe_name)
            if feedback:
                 print(f"\n[HITL] ✅ Approval Detected (Gmail)! Resuming...")
                 # Move file programmatically
                 if self.approve_tool(safe_name):
                     return f"Approved via Email. Feedback: {feedback}"
            
            # Wait
            time.sleep(30)

    def _notify_user(self, tool_name, desc, reason):
        print(f"\n[HITL] 📧 Tentativo invio notifica a {self.recipient_email} via Gmail API...")

        subject = f"🤖 [AGENTE AI] Richiesta Approvazione: {tool_name}"
        body = f"""
        L'agente richiede il permesso di creare un nuovo tool.
        
        - NOME: {tool_name}
        - MOTIVO: {reason}
        - DESCRIZIONE: {desc}
        
        STATO: In attesa attiva (Polling). Il job SLURM non è terminato.
        
        Per approvare e sbloccare l'agente, esegui:
        python cluster/scripts/approve_tool.py {tool_name}
        """

        result = self.google_client.send_email(self.recipient_email, subject, body)
        
        if result and "error" not in result:
             print("[HITL] ✅ Email inviata con successo via Gmail API.")
        else:
             print(f"[HITL] ⚠️ Impossibile inviare email (Errore: {result.get('error')})")
             self._log_backup_notification(tool_name, reason)

    def _log_backup_notification(self, tool_name, reason):
        print(f"\n{'='*60}")
        print(f"FALLBACK LOG NOTIFICATION")
        print(f"Tool: {tool_name}")
        print(f"Reason: {reason}")
        print(f"Action: Esegui 'python cluster/scripts/approve_tool.py {tool_name}'")
        print(f"{'='*60}\n")

    def approve_tool(self, tool_name):
        """Sposta il tool da Pending a Active."""
        safe_name = tool_name.replace(".py", "")
        src = os.path.join(STAGING_DIR, f"{safe_name}.py")
        dst = os.path.join(ACTIVE_DIR, f"{safe_name}.py")
        meta = os.path.join(STAGING_DIR, f"{safe_name}.json")

        if os.path.exists(src):
            shutil.move(src, dst)
            if os.path.exists(meta): os.remove(meta)
            print(f"[HITL] ✅ Tool '{safe_name}' APPROVATO e ATTIVATO in {ACTIVE_DIR}.")
            return True
        else:
            print(f"[HITL] ❌ Errore: Tool '{safe_name}' non trovato in {STAGING_DIR}.")
            return False
