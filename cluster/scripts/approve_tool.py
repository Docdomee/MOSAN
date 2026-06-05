import sys
import os
import argparse
import json

# Path hack per importare cluster.core
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from cluster.core.human_interface import HumanInterface
except ImportError:
     print("Error: Could not import HumanInterface. Ensure you are running from the project root.")
     sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Approve a pending tool for the Cluster Agent.")
    parser.add_argument("tool_name", help="Name of the tool to approve")
    parser.add_argument("--message", help="Optional message/instruction for the agent upon resume", default=None)
    
    args = parser.parse_args()
    tool_name = args.tool_name

    staging = os.path.join(project_root, "cluster", "pending_tools")
    
    # Validation
    if not os.path.exists(os.path.join(staging, f"{tool_name}.py")):
        print(f"Error: Tool '{tool_name}' not found in {staging}.")
        # List available
        if os.path.exists(staging) and os.listdir(staging):
            print("Pending tools:")
            for f in os.listdir(staging):
                if f.endswith(".py"): print(f" - {f[:-3]}")
        return

    # Approve
    if HumanInterface().approve_tool(tool_name):
        # Save Feedback if provided
        if args.message:
            feedback_file = os.path.join(project_root, "cluster", "approval_feedback.json")
            with open(feedback_file, "w") as f:
                json.dump({
                    "tool_name": tool_name,
                    "message": args.message,
                    "status": "approved"
                }, f)
            print(f"[HITL] Feedback saved. Agent will see: '{args.message}'")
        else:
            # Save default approved status so agent knows it's a resume
            feedback_file = os.path.join(project_root, "cluster", "approval_feedback.json")
            with open(feedback_file, "w") as f:
                json.dump({
                    "tool_name": tool_name,
                    "message": "Approved by user.",
                    "status": "approved"
                }, f)

if __name__ == "__main__":
    main()
