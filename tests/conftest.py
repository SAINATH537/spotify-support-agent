"""tests/conftest.py — pytest configuration"""
import sys
from pathlib import Path

# Ensure src/ is importable in tests
sys.path.insert(0, str(Path(__file__).parent.parent))
