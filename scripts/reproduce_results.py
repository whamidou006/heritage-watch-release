"""Regenerate section 5 through the same package entry point as the CLI."""
import sys

from heritage_watch.cli import main


if __name__ == "__main__":
    main(["reproduce", *sys.argv[1:]])
