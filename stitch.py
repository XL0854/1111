"""Train, test, or compare the self-contained fixed-half stitching project."""

from pathlib import Path
import os
import runpy
import sys


def main():
    root = Path(__file__).resolve().parent
    codes = root / "codes"
    if len(sys.argv) < 2 or sys.argv[1] not in ("train", "test", "compare"):
        raise SystemExit("Usage: python stitch.py {train,test,compare} --help")
    command, *options = sys.argv[1:]
    if any(arg.split("=", 1)[0] in {"--rop", "--pixel", "--plane-mode"} for arg in options):
        raise SystemExit("This entry point uses the bundled sources and fixed-half plane.")
    options += ["--rop", str(root / "vendor/ropstitch"),
                "--pixel", str(root / "vendor/pixelstitch")]
    if command == "compare":
        os.environ.setdefault("TORCH_HOME", str(root / "vendor/ropstitch/torch_cache"))
        entry = codes / "compare_full_test.py"
    else:
        entry = codes / "full.py"
        options = [command, "--plane-mode", "fixed-half", *options]
    sys.argv = [str(entry), *options]
    sys.path.insert(0, str(codes))
    runpy.run_path(str(entry), run_name="__main__")


if __name__ == "__main__":
    main()
