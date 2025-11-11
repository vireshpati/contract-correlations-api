"""Verify code structure without requiring dependencies."""
import os
import sys
from pathlib import Path

print("=" * 80)
print("Contract Correlation API - Code Verification")
print("=" * 80)

# Check directory structure
print("\n1. Checking directory structure...")
required_dirs = [
    "src",
    "src/db",
    "src/ml",
    "tests",
    "examples",
    "resources",
]

for dir_path in required_dirs:
    if os.path.isdir(dir_path):
        print(f"  ✓ {dir_path}/")
    else:
        print(f"  ✗ {dir_path}/ - MISSING")

# Check required files
print("\n2. Checking required files...")
required_files = [
    "src/__init__.py",
    "src/main.py",
    "src/models.py",
    "src/config.py",
    "src/database.py",
    "src/db/__init__.py",
    "src/db/schema.py",
    "src/db/queries.py",
    "src/ml/__init__.py",
    "src/ml/inference.py",
    "src/ml/training.py",
    "src/ml/rag.py",
    "src/ml/prompt_builder.py",
    "tests/__init__.py",
    "tests/conftest.py",
    "tests/test_api.py",
    "tests/test_prompt_builder.py",
    "tests/test_queries.py",
    "examples/predict_correlation.py",
    "examples/train_model.py",
    "examples/test_database.py",
    "requirements.txt",
    ".env",
    "README.md",
]

for file_path in required_files:
    if os.path.isfile(file_path):
        size = os.path.getsize(file_path)
        print(f"  ✓ {file_path} ({size} bytes)")
    else:
        print(f"  ✗ {file_path} - MISSING")

# Verify Python syntax
print("\n3. Verifying Python syntax...")
python_files = []
for root, dirs, files in os.walk("src"):
    for file in files:
        if file.endswith(".py"):
            python_files.append(os.path.join(root, file))

for root, dirs, files in os.walk("tests"):
    for file in files:
        if file.endswith(".py"):
            python_files.append(os.path.join(root, file))

syntax_errors = 0
for py_file in python_files:
    try:
        with open(py_file, 'r') as f:
            compile(f.read(), py_file, 'exec')
        print(f"  ✓ {py_file}")
    except SyntaxError as e:
        print(f"  ✗ {py_file} - SYNTAX ERROR: {e}")
        syntax_errors += 1

# Count lines of code
print("\n4. Code statistics...")
total_lines = 0
for py_file in python_files:
    with open(py_file, 'r') as f:
        lines = len(f.readlines())
        total_lines += lines

print(f"  Total Python files: {len(python_files)}")
print(f"  Total lines of code: {total_lines}")

# Summary
print("\n" + "=" * 80)
if syntax_errors == 0:
    print("✓ All Python files have valid syntax!")
    print("\nNext steps:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Run tests: pytest tests/ -v")
    print("  3. Test database: python examples/test_database.py")
    print("  4. Start API: uvicorn src.main:app --reload")
else:
    print(f"✗ Found {syntax_errors} file(s) with syntax errors")
print("=" * 80)
