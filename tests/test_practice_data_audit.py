"""Проверка сравнения выборок и кэша масок без PyTorch и медицинских данных."""

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

from kidney_vessels.application.audit import AuditConfig, run_audit


def test_practice_audit_detects_duplicates_and_cache_mismatch(
    dataset: Path, tmp_path: Path
) -> None:
    metadata = dataset / "tile_meta.csv"
    metadata.write_text(
        metadata.read_text(encoding="utf-8").replace("b,1,2,32,0", "b,2,2,32,0"),
        encoding="utf-8",
    )
    cache = dataset / "mask_cache"
    cache.mkdir()
    for image_id in ("a", "b"):
        Image.new("L", (32, 32)).save(cache / f"{image_id}.png")
    report = tmp_path / "audit"
    run_audit(AuditConfig(dataset, report, examples=0))
    output = tmp_path / "split.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "audit_practice_data.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input",
            str(dataset),
            "--audit",
            str(report),
            "--output",
            str(output),
            "--train-wsi",
            "1",
            "--val-wsi",
            "2",
        ],
        capture_output=True,
        encoding="utf-8",
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["train"]["tiles"] == result["validation"]["tiles"] == 1
    assert result["cross_split"]["wsi"] == []
    assert result["cross_split"]["identical_rgb_pairs"] == 1
    assert result["cross_split"]["patients"] is None
    assert result["train"]["cache"]["mismatch"] == 1
    assert result["validation"]["cache"]["match"] == 1
    assert result["validation"]["masks_without_vessel_pixels"] == 1


def test_practice_audit_rejects_registry_with_changed_wsi(dataset: Path, tmp_path: Path) -> None:
    report = tmp_path / "audit"
    run_audit(AuditConfig(dataset, report, examples=0))
    images = report / "images.csv"
    original = images.read_text(encoding="utf-8-sig")
    assert ",RGB,1," in original
    images.write_text(original.replace(",RGB,1,", ",RGB,99,", 1), encoding="utf-8-sig")
    script = Path(__file__).resolve().parents[1] / "scripts" / "audit_practice_data.py"
    output = tmp_path / "split.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            str(script),
            "--input",
            str(dataset),
            "--audit",
            str(report),
            "--output",
            str(output),
            "--train-wsi",
            "1",
            "--val-wsi",
            "6",
        ],
        capture_output=True,
        encoding="utf-8",
    )
    assert completed.returncode != 0
    assert "не соответствует исходной tile_meta.csv" in completed.stderr
    assert not output.exists()
