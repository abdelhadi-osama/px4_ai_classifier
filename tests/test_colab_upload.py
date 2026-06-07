import os
import sys

def verify_colab_upload():
    print("--- Verifying Colab Upload Integrity ---")
    
    # Check for core directories
    required_dirs = ['src', 'data', 'src/models', 'src/dataloader']
    missing_dirs = [d for d in required_dirs if not os.path.isdir(d)]
    
    # Check for core files
    required_files = ['src/config.py', 'src/train.py', 'data/train_scaled.csv']
    missing_files = [f for f in required_files if not os.path.isfile(f)]
    
    if not missing_dirs and not missing_files:
        print("✅ SUCCESS: Project structure is intact.")
        print(f"Current Directory: {os.getcwd()}")
        print(f"Files in data/: {os.listdir('data')}")
        return True
    else:
        if missing_dirs:
            print(f"❌ MISSING DIRECTORIES: {missing_dirs}")
        if missing_files:
            print(f"❌ MISSING FILES: {missing_files}")
        return False

if __name__ == "__main__":
    if verify_colab_upload():
        sys.exit(0)
    else:
        sys.exit(1)
