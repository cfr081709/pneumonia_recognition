import subprocess
import sys
import os

BASE = os.path.dirname(os.path.abspath(__file__))

def run(script_name):
    script_path = os.path.join(BASE, script_name)
    print(f"\n{'='*50}")
    print(f"Running {script_name}...")
    print(f"{'='*50}\n")
    result = subprocess.run([sys.executable, script_path])
    if result.returncode != 0:
        print(f"\n  {script_name} failed. Stopping pipeline.")
        sys.exit(result.returncode)
    print(f"\n {script_name} complete.")

if __name__ == '__main__':
    run("train.py")
    run("eval.py")
    run("test.py")