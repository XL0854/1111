"""Train, test, or compare the self-contained fixed-half stitching project."""

from pathlib import Path
import os
import runpy
import sys

from project_config import DEPTH_DATA, TEST_DATA, TRAIN_DATA, require_directory


def has_option(options, name):
    return any(option.split("=", 1)[0] == name for option in options)


def main():
    root = Path(__file__).resolve().parent
    codes = root / "codes"
    if len(sys.argv) < 2 or sys.argv[1] not in ("train", "test", "compare", "patch-compare"):
        raise SystemExit("Usage: python stitch.py {train,test,compare,patch-compare} --help")
    command, *options = sys.argv[1:]
    if not has_option(options, "--data"):
        data = TRAIN_DATA if command == "train" else TEST_DATA
        options += ["--data", str(require_directory(data, f"{command} data"))]
    if command == "train" and not has_option(options, "--depth-data"):
        options += ["--depth-data", str(require_directory(DEPTH_DATA, "depth cache"))]
        if not has_option(options, "--depth-weight"):
            options += ["--depth-weight", "0.03"]
    patch_test = command == "test" and "--local-patch" in options
    if patch_test:
        options.remove("--local-patch")
        if "--metrics-only" in options:
            options.remove("--metrics-only")
        elif "--save-stitched" not in options:
            options.append("--save-stitched")
    if any(arg.split("=", 1)[0] in {"--rop", "--pixel", "--plane-mode"} for arg in options):
        raise SystemExit("This entry point uses the bundled sources and fixed-half plane.")
    options += ["--rop", str(root / "vendor/ropstitch"),
                "--pixel", str(root / "vendor/pixelstitch")]
    if command == "compare":
        os.environ.setdefault("TORCH_HOME", str(root / "vendor/ropstitch/torch_cache"))
        entry = codes / "compare_full_test.py"
    elif command == "patch-compare" or patch_test:
        entry = codes / "local_patch_align.py"
    else:
        entry = codes / "full.py"
        options = [command, "--plane-mode", "fixed-half", *options]
    sys.argv = [str(entry), *options]
    sys.path.insert(0, str(codes))
    runpy.run_path(str(entry), run_name="__main__")


if __name__ == "__main__":
    main()
