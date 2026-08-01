import importlib
import sys

sys.modules.setdefault("traffic_adv", importlib.import_module("src.traffic_adv"))
from src.traffic_adv.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["attack", *__import__("sys").argv[1:]]))
