import dotenv
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # Ensure script dir is in sys.path
import argparse
# Import the function that creates the flow
from flow import create_tutorial_flow

dotenv.load_dotenv()

# Default file patterns
DEFAULT_INCLUDE_PATTERNS = {
    "*.py", "*.js", "*.jsx", "*.ts", "*.tsx", "*.go", "*.java", "*.pyi", "*.pyx",
    "*.c", "*.cc", "*.cpp", "*.h", "*.md", "*.rst", "Dockerfile",
    "Makefile", "*.yaml", "*.yml",
}

DEFAULT_EXCLUDE_PATTERNS = {
    "venv/*", ".venv/*", "*test*", "tests/*", "docs/*", "examples/*", "v1/*",
    "dist/*", "build/*", "experimental/*", "deprecated/*",
    "legacy/*", ".git/*", ".github/*", ".next/*", ".vscode/*", "obj/*", "bin/*", "node_modules/*", "*.log"
}

# --- Main Function ---
def main():
    parser = argparse.ArgumentParser(description="Generate a tutorial for a GitHub codebase or local directory.")

    # Create mutually exclusive group for source
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--repo", help="URL of the public GitHub repository.")
    source_group.add_argument("--dir", help="Path to local directory.")
    source_group.add_argument("--file", help="Path to a single input file (e.g., PDF or other document).")
    source_group.add_argument("--url", help="URL to a web page or document.")

    parser.add_argument("-n", "--name", help="Project name (optional, derived from repo/directory if omitted).")
    parser.add_argument("-t", "--token", help="GitHub personal access token (optional, reads from GITHUB_TOKEN env var if not provided).")
    parser.add_argument("-o", "--output", default="output", help="Base directory for output (default: ./output).")
    parser.add_argument("-i", "--include", nargs="+", help="Include file patterns (e.g. '*.py' '*.js'). Defaults to common code files if not specified.")
    parser.add_argument("-e", "--exclude", nargs="+", help="Exclude file patterns (e.g. 'tests/*' 'docs/*'). Defaults to test/build directories if not specified.")
    parser.add_argument("-s", "--max-size", type=int, default=100000, help="Maximum file size in bytes (default: 100000, about 100KB).")
    # Add language parameter for multi-language support
    parser.add_argument("--language", default="english", help="Language for the generated tutorial (default: english)")

    args = parser.parse_args()

    # If --url is provided, fetch the content and save to a temp file, then set args.file
    if args.url:
        import tempfile
        import os
        try:
            try:
                import requests
                resp = requests.get(args.url)
                resp.raise_for_status()
                content = resp.text
            except ImportError:
                from urllib.request import urlopen
                resp = urlopen(args.url)
                content = resp.read().decode('utf-8')
            # Write content to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.txt', mode='w', encoding='utf-8') as tmpf:
                tmpf.write(content)
                tmp_file_path = tmpf.name
            print(f"Downloaded URL content to {tmp_file_path}")
            args.file = tmp_file_path
        except Exception as e:
            print(f"Failed to fetch URL {args.url}: {e}")
            exit(1)

    # Get GitHub token from argument or environment variable if using repo
    github_token = None
    if args.repo:
        github_token = args.token or os.environ.get('GITHUB_TOKEN')
        if not github_token:
            print("Warning: No GitHub token provided. You might hit rate limits for public repositories.")

    # Initialize the shared dictionary with inputs
    shared = {
        "repo_url": args.repo,
        "single_file": args.file,
        "local_dir": args.dir,
        "project_name": args.name, # Can be None, FetchRepo will derive it
        "github_token": github_token,
        "output_dir": args.output, # Base directory for CombineTutorial output

        # Add include/exclude patterns and max file size
        "include_patterns": set(args.include) if args.include else DEFAULT_INCLUDE_PATTERNS,
        "exclude_patterns": set(args.exclude) if args.exclude else DEFAULT_EXCLUDE_PATTERNS,
        "max_file_size": args.max_size,

        # Add language for multi-language support
        "language": args.language,

        # Outputs will be populated by the nodes
        "files": [],
        "abstractions": [],
        "relationships": {},
        "chapter_order": [],
        "chapters": [],
        "final_output_dir": None
    }

    # Display starting message with repository/directory and language
    print(f"Starting tutorial generation for: {args.repo or args.dir} in {args.language.capitalize()} language")

    # Create the flow instance
    tutorial_flow = create_tutorial_flow()

    # Run the flow with enhanced error handling
    try:
        tutorial_flow.run(shared)
    except Exception as e:
        import traceback
        print(f"\nERROR: {type(e).__name__}: {e}")
        print("\nDetailed traceback:")
        traceback.print_exc()
        
        # Print additional debugging info
        print("\nDebugging information:")
        if hasattr(e, 'response'):
            print(f"Response info: {getattr(e, 'response', None)}")
        
        print("\nShared context:")
        for key, value in shared.items():
            if isinstance(value, (str, int, float, bool)):
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {type(value).__name__} object")
        
        # Exit with error code
        exit(1)

if __name__ == "__main__":
    main()
