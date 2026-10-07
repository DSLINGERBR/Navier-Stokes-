import json, pathlib, sys
import pytest
RAIZ = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

@pytest.fixture(scope="session")
def dados():
    return {p.stem: json.loads(p.read_text()) for p in (RAIZ / "dados").glob("*.json")}
