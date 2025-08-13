# Patch flask_uploads.py to fix ImportError with werkzeug
import os
venv_path = '/Users/thealchemist/Documents/GitHub/HyperbolicLearner/.venv/lib/python3.10/site-packages/flask_uploads.py'
if os.path.exists(venv_path):
    with open(venv_path, 'r') as f:
        lines = f.readlines()
    with open(venv_path, 'w') as f:
        for line in lines:
            if 'from werkzeug import secure_filename' in line:
                f.write(line.replace('from werkzeug import secure_filename', 'from werkzeug.utils import secure_filename'))
            else:
                f.write(line)
print('Patched flask_uploads.py if needed.')
