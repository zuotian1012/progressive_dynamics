import argparse

from .simulation import METHODS, run
from .report import generate


def main():
    parser = argparse.ArgumentParser(description="Progressive triangular-shell cloth experiments")
    parser.add_argument("--config", default="configs/hanging.json")
    parser.add_argument("--output", default="output/hanging")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args()
    run(args.config, args.output, args.methods)
    generate(args.output, animate=not args.no_gif)


if __name__ == "__main__":
    main()
