# -*- coding: utf-8 -*-

import json
import csv
from pathlib import Path


JSONL_PATHS = [
     r"C:\D drive\Cornell_Term\AU25\CS 5470\project\plots\deepseek\deepseek_ep1.jsonl",
     r"C:\D drive\Cornell_Term\AU25\CS 5470\project\plots\deepseek\deepseek_ep2.jsonl",
     r"C:\D drive\Cornell_Term\AU25\CS 5470\project\plots\deepseek\deepseek_ep4.jsonl",
]

CSV_OUT =  r"C:\D drive\Cornell_Term\AU25\CS 5470\project\plots\deepseek\throughput_summary.csv"


def load_single_record(jsonl_path: Path) -> dict:
    record = None
    count = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            count += 1
            if count == 1:
                record = json.loads(line)
            else:
                raise ValueError(
                    f"{jsonl_path}: expected exactly 1 record, found {count}"
                )
    if record is None:
        raise ValueError(f"{jsonl_path}: no record found")
    return record



if not JSONL_PATHS:
    raise ValueError("JSONL_PATHS is empty — add paths to your .jsonl files")

paths = [Path(p) for p in JSONL_PATHS]
for p in paths:
    if not p.exists():
        raise FileNotFoundError(p)

rows = []

for p in paths:
    r = load_single_record(p)

    duration = float(r["duration"])   # seconds
    completed = int(r["completed"])

    in_tok = int(r.get("total_input_tokens", 0))
    out_tok = int(r.get("total_output_tokens", 0))
    total_tok = in_tok + out_tok

    rows.append({
        "run": p.stem,
        "duration_s": duration,
        "completed": completed,
        "requests_per_sec": completed / duration if duration > 0 else None,
        "input_tokens_per_sec": in_tok / duration if duration > 0 else None,
        "output_tokens_per_sec": out_tok / duration if duration > 0 else None,
        "total_tokens_per_sec": total_tok / duration if duration > 0 else None,
        "json_request_throughput": r.get("request_throughput"),
        "json_total_throughput": r.get("total_throughput"),
    })

# ==========================
# Write CSV
# ==========================

csv_path = Path(CSV_OUT)
csv_path.parent.mkdir(parents=True, exist_ok=True)

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"CSV written to: {csv_path.resolve()}")

# Optional: print summary to console
for row in rows:
    print(row)
