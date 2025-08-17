import sys
import site
import os

# Add user site-packages to path if not already there
user_site = site.getusersitepackages()
if user_site not in sys.path:
    sys.path.append(user_site)
    print(f"Added user site-packages: {user_site}")
else:
    print(f"User site-packages already in path: {user_site}")

# Print current Python executable and path
print(f"Python executable: {sys.executable}")
print("Python path:")
for p in sys.path:
    print(f"  {p}")

# Check if tkcalendar can be imported
try:
    import tkcalendar
    print(f"tkcalendar found at: {tkcalendar.__file__}")
    print(f"tkcalendar version: {tkcalendar.__version__}")
except ImportError as e:
    print(f"Error importing tkcalendar: {e}")
