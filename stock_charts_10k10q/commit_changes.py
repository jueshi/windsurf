import os
import subprocess
import sys

def run_command(command):
    """Run a command and return its output"""
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print(f"Stderr: {e.stderr}")
        return False

def commit_changes():
    """Initialize git repo and commit all changes"""
    # Check if .git directory exists
    if not os.path.exists('.git'):
        print("Initializing Git repository...")
        if not run_command('git init'):
            return False
    
    # Configure Git if needed
    if not run_command('git config user.name "Stock Charts User"'):
        return False
    if not run_command('git config user.email "user@example.com"'):
        return False
    
    # Add all files
    print("Adding files to Git...")
    if not run_command('git add .'):
        return False
    
    # Commit changes
    print("Committing changes...")
    commit_message = "Fix tab switching issue with Extract 10-K Tables button"
    if not run_command(f'git commit -m "{commit_message}"'):
        return False
    
    print("Changes committed successfully!")
    return True

if __name__ == "__main__":
    # Change to the script's directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Commit changes
    if commit_changes():
        print("All operations completed successfully.")
    else:
        print("Failed to commit changes.")
        sys.exit(1)
