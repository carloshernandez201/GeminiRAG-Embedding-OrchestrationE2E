import sys
from pathlib import Path

# The app modules import each other as top-level modules (as they do in Lambda)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
