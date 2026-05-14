import csv
import shutil
from pathlib import Path

CSV_PATH = Path("non-leaky/splits/test.csv")
EXPORT_DIR = Path("clean_test_export")

EXPORT_DIR.mkdir(exist_ok=True)

count = 0

with open(CSV_PATH) as f:
    reader = csv.DictReader(f)

    for row in reader:
        src = Path(row["path"])

        cls = row["class_name"]
        dst_dir = EXPORT_DIR / cls
        dst_dir.mkdir(parents=True, exist_ok=True)

        dst = dst_dir / src.name

        shutil.copy2(src, dst)
        count += 1

print("Exported:", count)