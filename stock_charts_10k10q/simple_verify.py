"""
Simple verification of SEC API caching and retry logic
"""
import os
import time
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check if SEC email is set
sec_email = os.getenv("SEC_EDGAR_EMAIL")
print(f"SEC_EDGAR_EMAIL: {'Set' if sec_email else 'Not set'}")

def check_sec_cache_dir():
    """Check if SEC cache directory exists and show its contents"""
    print("\n=== Checking SEC Cache Directory ===")
    
    sec_cache_dir = Path("sec_cache")
    if not sec_cache_dir.exists():
        print(f"SEC cache directory not found: {sec_cache_dir}")
        return False
    
    print(f"SEC cache directory exists: {sec_cache_dir}")
    
    # List subdirectories
    subdirs = [d for d in sec_cache_dir.iterdir() if d.is_dir()]
    print(f"Found {len(subdirs)} subdirectories: {[d.name for d in subdirs]}")
    
    # Check CIK cache directory
    cik_cache_dir = sec_cache_dir / "cik_lookups"
    if cik_cache_dir.exists():
        print(f"CIK cache directory exists: {cik_cache_dir}")
        cik_files = list(cik_cache_dir.glob("*.txt"))
        print(f"Found {len(cik_files)} CIK cache files")
        
        # Show a few sample files
        for file in cik_files[:5]:
            try:
                with open(file, 'r') as f:
                    cik = f.read().strip()
                    print(f"Ticker {file.stem} -> CIK: {cik}")
            except Exception as e:
                print(f"Error reading CIK file {file}: {e}")
    
    return True

# Run the check
check_sec_cache_dir()
