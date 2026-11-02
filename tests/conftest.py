"""Небольшой синтетический набор с контрактом исходных файлов HuBMAP."""

import csv
import json
from collections.abc import Callable
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "hubmap"
    (root / "train").mkdir(parents=True)
    for image_id in ("a", "b", "c"):
        Image.new("RGB", (32, 32), (220, 185, 205)).save(root / "train" / f"{image_id}.tif")
    with (root / "tile_meta.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "source_wsi", "dataset", "i", "j"])
        writer.writerows(
            [["a", "1", "1", "0", "0"], ["b", "1", "2", "32", "0"], ["c", "6", "3", "0", "0"]]
        )
    (root / "wsi_meta.csv").write_text(
        "source_wsi,age,sex,race,height,weight,bmi\n1,58,F,W,160,59,23\n", encoding="utf-8"
    )
    records = [
        {
            "id": "a",
            "annotations": [
                {"type": "blood_vessel", "coordinates": [[[2, 2], [12, 2], [12, 12], [2, 12]]]},
                {"type": "unsure", "coordinates": [[[16, 16], [20, 16], [20, 20], [16, 20]]]},
                {"type": "glomerulus", "coordinates": [[[22, 2], [28, 2], [28, 8], [22, 8]]]},
            ],
        },
        {"id": "b", "annotations": []},
    ]
    (root / "polygons.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    return root


@pytest.fixture
def set_annotations(dataset: Path) -> Callable[[str], Path]:
    def write(content: str) -> Path:
        path = dataset / "polygons.jsonl"
        path.write_text(content, encoding="utf-8")
        return path

    return write
