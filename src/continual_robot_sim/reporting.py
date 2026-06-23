from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def save_matrix_csv(matrix: np.ndarray, task_names: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["after_task"] + task_names)
        for i, task_name in enumerate(task_names):
            row = [task_name]
            for value in matrix[i]:
                row.append("" if np.isnan(value) else f"{value:.4f}")
            writer.writerow(row)


def save_json(data: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def forgetting_summary(matrix: np.ndarray, task_names: list[str]) -> dict[str, float]:
    summary = {}
    final_row = matrix[-1]
    for index, task_name in enumerate(task_names):
        column = matrix[:, index]
        best = float(np.nanmax(column))
        final = float(final_row[index])
        summary[task_name] = best - final
    return summary
