import sys
import platform

print(f"Python Version: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"Python Executable: {sys.executable}")

try:
    import pip
    print(f"Pip Version: {pip.__version__}")
except ImportError:
    print("Pip not found")
