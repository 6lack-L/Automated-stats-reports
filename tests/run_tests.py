import traceback
import sys
import os
import importlib.util
from glob import glob

# Ensure cwd is on sys.path so imports inside test modules (e.g., build_workbook) resolve
cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)


def discover_tests():
    """Discover and load all test functions from tests/test_*.py files."""
    tests = []
    tests_dir = os.path.join(cwd, "tests")
    pattern = os.path.join(tests_dir, "test_*.py")
    for path in sorted(glob(pattern)):
        mod_name = os.path.splitext(os.path.basename(path))[0]
        try:
            spec = importlib.util.spec_from_file_location(mod_name, path)
            if spec is None or spec.loader is None:
                print(f"ERROR: no spec/loader for {path}")
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        except Exception as e:
            print(f"ERROR: failed to import {path}: {e}")
            traceback.print_exc()
            continue
        for name in dir(mod):
            if name.startswith("test_"):
                fn = getattr(mod, name)
                if callable(fn):
                    tests.append(fn)
    return tests


def run_one(fn):
    try:
        fn()
        print(f"PASS: {fn.__name__}")
        return True
    except AssertionError as e:
        print(f"FAIL: {fn.__name__} - AssertionError: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"ERROR: {fn.__name__} - Exception: {e}")
        traceback.print_exc()
        return False


def main():
    tests = discover_tests()
    ok = True
    for t in tests:
        ok = run_one(t) and ok
    if not ok:
        print("Some tests failed")
        sys.exit(2)
    print("All tests passed")


if __name__ == '__main__':
    main()
