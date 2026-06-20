import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model import Transformer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs.calliope_10m")
    args = parser.parse_args()
    cfg = importlib.import_module(args.config).model_config
    model = Transformer(cfg)
    print(f"{model.num_parameters():,} parameters")


if __name__ == "__main__":
    main()
