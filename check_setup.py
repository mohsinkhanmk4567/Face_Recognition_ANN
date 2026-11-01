"""
check_setup.py
==============
Run this once after installing the requirements:

    python check_setup.py

It checks Python, creates the project folders, creates the SQLite database
and tries to import every library. It does NOT test the webcam or the
neural network (those come in later stages).
"""

import importlib
import sys

from config import settings
from database.database import Database

LIBRARIES = [
    # (import name, pip name)
    ("tensorflow", "tensorflow"),
    ("cv2", "opencv-python"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("sklearn", "scikit-learn"),
    ("matplotlib", "matplotlib"),
    ("customtkinter", "customtkinter"),
    ("PIL", "Pillow"),
    ("tkinter", "tkinter (reinstall Python from python.org, tick 'tcl/tk')"),
]


def main() -> int:
    problems = 0

    # 1. Python version
    v = sys.version_info
    bits = 64 if sys.maxsize > 2 ** 32 else 32
    print(f"Python {v.major}.{v.minor}.{v.micro} ({bits}-bit)")
    if (v.major, v.minor) not in ((3, 10), (3, 11), (3, 12)) or bits != 64:
        print("  [WARN] TensorFlow needs 64-bit Python 3.10, 3.11 or 3.12 "
              "(3.12 recommended). Newer versions such as 3.15 have no "
              "TensorFlow wheels yet.")
        problems += 1

    # 2. Settings and folders
    settings.validate_settings()
    settings.ensure_directories()
    print("Settings valid, folders ready:", settings.BASE_DIR)

    # 3. Database
    db = Database()
    db.initialize()
    print(f"Database ready: {settings.DB_PATH} "
          f"(schema v{db.schema_version()}, {db.count_users()} users)")

    # 4. Libraries
    print("Libraries:")
    for import_name, pip_name in LIBRARIES:
        try:
            module = importlib.import_module(import_name)
            version = getattr(module, "__version__", "")
            print(f"  [OK]      {import_name} {version}")
        except Exception as exc:                      # missing or broken install
            print(f"  [MISSING] {import_name} -> install: {pip_name} ({exc})")
            problems += 1

    print()
    if problems:
        print(f"Setup check finished with {problems} problem(s). Fix them and run again.")
        return 1
    print("Setup check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
