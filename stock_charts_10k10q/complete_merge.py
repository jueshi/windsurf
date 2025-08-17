import subprocess
import os
import sys

def run_command(command):
    """Run a command and return its output"""
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, f"Error: {e.stderr}"

def complete_merge():
    """Complete the Git merge process"""
    print("Starting merge completion process...")
    
    # Step 1: Add all files except .env
    success, output = run_command("git add .")
    if not success:
        print(f"Failed to add files: {output}")
        return False
    
    # Step 2: Reset .env to ensure it's not included in the commit
    success, output = run_command("git reset .env")
    if not success:
        print(f"Failed to reset .env: {output}")
        # Continue anyway, this might not be critical
    
    # Step 3: Commit the merge
    success, output = run_command('git commit -m "Merge remote branch with local changes, fix tab switching issues"')
    if not success:
        print(f"Failed to commit: {output}")
        return False
    
    # Step 4: Push to remote
    print("Attempting to push changes to remote...")
    success, output = run_command("git push origin master")
    if not success:
        print(f"Failed to push: {output}")
        print("You may need to push manually.")
        return False
    
    print("Merge completed successfully!")
    return True

if __name__ == "__main__":
    complete_merge()
