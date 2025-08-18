"""
Script to check for f-string syntax errors in gemini_analyzer.py
"""

def check_file_syntax(filename):
    """Check a Python file for f-string syntax errors."""
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        if 'f"' in line or "f'" in line:
            if '\\n' in line and '{' in line and '}' in line:
                print(f"Line {i+1}: {line.strip()}")
                # Also print surrounding lines for context
                if i > 0:
                    print(f"Line {i}: {lines[i-1].strip()}")
                if i < len(lines) - 1:
                    print(f"Line {i+2}: {lines[i+1].strip()}")
                print("-" * 50)

if __name__ == "__main__":
    check_file_syntax("gemini_analyzer.py")
