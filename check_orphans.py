import os
import glob
import re

all_py = glob.glob('*.py')
all_py += glob.glob('cluster/*.py')
all_py += glob.glob('processed_data/scripts_worker/*.py')

def check_orphan(file_path):
    name = os.path.basename(file_path)[:-3]
    if name in ['main', 'agent', '__init__', 'celery_app', 'monitor_training']:
        return False
    
    # Simple check: does the string "name" appear anywhere in the codebase?
    # Actually, check "import name" or "from name"
    found = False
    for root, dirs, files in os.walk('.'):
        if 'venv' in root or '.git' in root or '__pycache__' in root:
            continue
        for f in files:
            if not f.endswith('.py'): continue
            fp = os.path.join(root, f)
            if fp == os.path.join('.', file_path): continue
            
            with open(fp, 'r', encoding='utf-8', errors='ignore') as f_obj:
                content = f_obj.read()
                if re.search(r'\bimport ' + re.escape(name) + r'\b', content) or re.search(r'\bfrom ' + re.escape(name) + r'\b', content) or re.search(rf'[\'"]{re.escape(name)}[\'"]', content):
                    found = True
                    break
        if found:
            break
    return not found

orphans = []
for py in all_py:
    if check_orphan(py):
        orphans.append(py)

print("ORPHANS:", orphans)
