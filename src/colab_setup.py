# ==========================================
# COLAB SETUP GUIDE (Run this in a cell)
# ==========================================

"""
# 1. Mount Google Drive (Recommended)
from google.colab import drive
drive.mount('/content/drive')

# 2. Navigate to your project folder
%cd /content/drive/MyDrive/path_to_your_project

# 3. Install missing dependencies
!pip install -r pyproject.toml --quiet

# 4. Run the training with Python path set
!export PYTHONPATH=$PYTHONPATH:. && python3 src/train.py
"""

import sys
import os

def prepare_colab_environment():
    """
    Optional helper to fix paths if running directly in a Colab Notebook cell.
    """
    if 'google.colab' in sys.modules:
        print("Detected Google Colab environment.")
        # Ensure the project root is in the system path
        project_root = os.getcwd()
        if project_root not in sys.path:
            sys.path.append(project_root)
            print(f"Added {project_root} to PYTHONPATH")

# Add this to the top of src/train.py if you want it to be automatic
