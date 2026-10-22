import json
from pathlib import Path

from hindsight.main import app

CONTRACT = Path(__file__).parent.parent / "contract" / "openapi.json"


def render() -> str:
    return json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    CONTRACT.write_text(render())
    print(f"wrote {CONTRACT}")
