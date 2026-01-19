from pathlib import Path

def get_project_path() -> str:
    return str(Path(__file__).parent.parent.parent)
