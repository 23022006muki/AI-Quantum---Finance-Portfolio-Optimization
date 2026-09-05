from pathlib import Path
from src.pipeline import run

HERE = Path(__file__).resolve().parent
if __name__ == "__main__":
    out = run(HERE, HERE / "config.yaml", HERE / "outputs" / "latest")
    print(out.relative_to(HERE).as_posix())
