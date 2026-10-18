import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.cli import main  # noqa: E402
def run(cmd): raise SystemExit(main([cmd, *sys.argv[1:]]))
