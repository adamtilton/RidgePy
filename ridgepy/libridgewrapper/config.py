from ctypes import cdll
from pathlib import Path
import re
import ast

MACRO_PATTERN = re.compile(r"#define\s+(\w+)\s+(\(*[+-]?\ *\d+\.*\d*\)*)")

def extract_macros_from_header(header_file_path_string):
    header_file_path = Path(header_file_path_string)

    with open(header_file_path, "r") as fh:
        header_file_contents = fh.read()

    macro_dict = dict()
    for name, value in re.findall(MACRO_PATTERN, header_file_contents):
        macro_dict[name] = ast.literal_eval(value)

    return macro_dict

BASE_DIR     = Path(__file__).resolve().parent.parent
CONFIG_FILE  = BASE_DIR.joinpath("libridge/inc/libridge.h")
LIBRARY_FILE = BASE_DIR.joinpath("libridge/build/libs/libridge.so")

assert CONFIG_FILE.exists(), f"Could not find config file at {CONFIG_FILE}"
assert LIBRARY_FILE.exists(), f"Could not find library file at {LIBRARY_FILE}"

MAIN_CONFIG = extract_macros_from_header(CONFIG_FILE)
LIBRARY     = cdll.LoadLibrary(LIBRARY_FILE)

if __name__ == "__main__":
    print("MAIN_CONFIG:")
    for key, value in MAIN_CONFIG.items():
        print(f"{key}: {value}")