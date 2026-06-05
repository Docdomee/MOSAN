
import os
import sys

try:
    from state_manager import PERSISTENT_PATHS, PROJECT_ROOT
    print(f"CWD: {os.getcwd()}")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"Strategic Dir: {PERSISTENT_PATHS['strategic_memory_dir']}")
    
    strategic_dir = PERSISTENT_PATHS['strategic_memory_dir']
    if os.path.exists(strategic_dir):
        print(f"Dir Exists: YES")
        try:
             files = os.listdir(strategic_dir)
             print(f"Files: {files}")
        except Exception as e:
             print(f"Ls Error: {e}")
    else:
        print(f"Dir Exists: NO")

except Exception as e:
    print(f"Error: {e}")
