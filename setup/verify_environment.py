#!/usr/bin/env python3
"""
Verify that the Python environment is correctly set up for HADES-Lab
"""

import subprocess
import sys
from pathlib import Path


def check_python_version():
    """Check Python version meets requirements."""
    version = sys.version_info
    print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major == 3 and version.minor >= 11:
        print("  → Python version meets requirements (>=3.11)")
        return True
    else:
        print("  → WARNING: Python version should be 3.11 or higher")
        return False

def check_uv():
    """Check if uv is available."""
    try:
        result = subprocess.run(['uv', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ uv installed: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    print("✗ uv not found in PATH")
    return False

def check_poetry():
    """Check if Poetry is available."""
    try:
        result = subprocess.run(['poetry', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Poetry installed: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    print("✗ Poetry not found in PATH")
    return False

def check_virtual_env():
    """Check if running in a virtual environment."""
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )

    if in_venv:
        print(f"✓ Running in virtual environment: {sys.prefix}")
        return True
    else:
        print("! Not in virtual environment - run 'poetry shell' to activate")
        return False

def check_key_packages():
    """Check if key packages are importable."""
    packages = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('docling', 'Docling'),
        ('arango', 'ArangoDB client'),
        ('sentence_transformers', 'Sentence Transformers'),
        ('pydantic', 'Pydantic'),
    ]

    print("\nChecking key packages:")
    all_good = True
    for package, name in packages:
        try:
            __import__(package)
            print(f"  ✓ {name} ({package})")
        except ImportError as e:
            print(f"  ✗ {name} ({package}): {e}")
            all_good = False

    return all_good

def check_mcp_readiness():
    """Check if environment is ready for MCP servers."""
    print("\nMCP Server Readiness:")

    # Check if .venv exists
    venv_path = Path.cwd() / '.venv'
    if venv_path.exists():
        print(f"  ✓ Virtual environment found at: {venv_path}")
    else:
        print(f"  ✗ Virtual environment not found at: {venv_path}")
        return False

    # Check if uv can run Python scripts
    try:
        result = subprocess.run(
            ['uv', 'run', 'python', '-c', 'print("MCP ready")'],
            capture_output=True,
            text=True,
            cwd=Path.cwd()
        )
        if result.returncode == 0 and "MCP ready" in result.stdout:
            print("  ✓ uv can run Python scripts")
            return True
        else:
            print(f"  ✗ uv run failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"  ✗ Could not test uv run: {e}")
        return False

def main():
    """Run all environment checks."""
    print("=" * 60)
    print("HADES-Lab Environment Verification")
    print("=" * 60)

    checks = {
        "Python Version": check_python_version(),
        "uv Package Manager": check_uv(),
        "Poetry": check_poetry(),
        "Virtual Environment": check_virtual_env(),
        "Key Packages": check_key_packages(),
        "MCP Readiness": check_mcp_readiness(),
    }

    print("\n" + "=" * 60)
    print("Summary:")
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check}")

    if all(checks.values()):
        print("\n✓ Environment is fully configured!")
        print("\nYou can now:")
        print("  1. Run MCP servers with: uv run python your_mcp_server.py")
        print("  2. Run the ACID pipeline with: poetry run python arxiv/pipelines/arxiv_pipeline.py")
    else:
        print("\n! Some checks failed. Please address the issues above.")
        if not checks["Virtual Environment"]:
            print("\nTip: Activate the Poetry environment with: poetry shell")

    print("=" * 60)

if __name__ == "__main__":
    main()
