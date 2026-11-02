"""Полный аудит синтетических данных и контракт командной строки."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from kidney_vessels.application.audit import AuditConfig, run_audit
from kidney_vessels.data.hubmap import DatasetStructureError


def test_full_audit_preserves_annotation_semantics_and_reports(
    dataset: Path, tmp_path: Path
) -> None:
    output = tmp_path / "reports"
    result = run_audit(AuditConfig(dataset, output, examples=3))
    assert result.summary["image_files"] == 3
    assert result.summary["objects"] == 3
    assert result.summary["source_wsi_count"] == 2
    assert result.summary["categories"] == {"blood_vessel": 1, "glomerulus": 1, "unsure": 1}
    rows = {row["image_id"]: row for row in result.images}
    assert rows["a"]["annotation_status"] == "expert_reviewed"
    assert rows["a"]["annotation_completeness"] == "unknown"
    assert rows["b"]["annotation_completeness"] == "sparse"
    assert rows["b"]["annotation_presence"] == "empty"
    assert rows["c"]["annotation_status"] == "unannotated"
    assert rows["c"]["annotation_presence"] == "missing"
    assert rows["a"]["patient_id"] is None
    assert rows["a"]["scale"] is None
    assert json.loads((output / "summary.json").read_text(encoding="utf-8")) == result.summary
    assert (output / "audit.md").is_file()
    assert (output / "manifest.json").is_file()
    assert (output / "registry.json").is_file()
    for filename in ("images.csv", "annotations.csv", "issues.csv", "geometry_statistics.csv"):
        with (output / filename).open(encoding="utf-8-sig", newline="") as stream:
            assert csv.DictReader(stream).fieldnames
    with (output / "annotations.csv").open(encoding="utf-8-sig", newline="") as stream:
        annotations = list(csv.DictReader(stream))
    vessel = next(row for row in annotations if row["category"] == "blood_vessel")
    assert float(vessel["area_px2"]) == 100
    assert float(vessel["bbox_width_px"]) == 10
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["examples"] == 3
    assert len(manifest["input_files"]) == 6
    assert all(len(entry["sha256"]) == 64 for entry in manifest["input_files"])
    assert manifest["source_sha256"]["domain.py"]
    previews = list((output / "examples").glob("*.png"))
    assert len(previews) == 3
    with Image.open(output / str(rows["a"]["preview"])) as image:
        assert image.convert("RGB").getpixel((2, 2)) == (230, 30, 70)


def test_bad_records_corrupt_image_unknown_category_and_orphan_are_kept(
    dataset: Path, tmp_path: Path
) -> None:
    (dataset / "train" / "b.tif").write_bytes(b"broken")
    with (dataset / "polygons.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("{bad json}\n")
        stream.write(
            json.dumps(
                {
                    "id": "absent",
                    "annotations": [
                        {"type": "novel", "coordinates": [[[1, 1], [3, 1], [3, 3]]]},
                    ],
                }
            )
            + "\n"
        )
    result = run_audit(AuditConfig(dataset, tmp_path / "reports"))
    codes = {issue.code for issue in result.issues}
    assert {"corrupt_image", "invalid_json", "orphan_annotation", "unknown_category"} <= codes
    assert result.summary["objects"] == 4
    assert result.summary["measurable_objects"] == 3
    registry = json.loads((result.output / "registry.json").read_text(encoding="utf-8"))
    assert any("{bad json}" in record["raw_line"] for record in registry["source_records"])
    assert len(registry["annotations"]) == 4


def test_missing_metadata_and_invalid_coordinates_are_reported(
    dataset: Path, tmp_path: Path
) -> None:
    (dataset / "tile_meta.csv").write_text(
        "id,source_wsi,dataset,i,j\na,1,1,,x\n", encoding="utf-8"
    )
    (dataset / "polygons.jsonl").write_text(
        json.dumps(
            {
                "id": "a",
                "annotations": [
                    {"type": "blood_vessel", "coordinates": [[[0, 0], [100, 0], [1, 1]]]},
                ],
            }
        ),
        encoding="utf-8",
    )
    result = run_audit(AuditConfig(dataset, tmp_path / "reports", examples=0))
    assert {"missing_tile_metadata", "invalid_tile_offset", "coordinate_out_of_bounds"} <= {
        issue.code for issue in result.issues
    }
    assert result.summary["measurable_objects"] == 0
    assert not (result.output / "examples").exists()


@pytest.mark.parametrize("filename", ["train", "polygons.jsonl", "tile_meta.csv", "wsi_meta.csv"])
def test_missing_required_path_has_clear_error(
    dataset: Path, tmp_path: Path, filename: str
) -> None:
    (dataset / filename).rename(dataset / (filename + ".missing"))
    with pytest.raises(DatasetStructureError, match=filename):
        run_audit(AuditConfig(dataset, tmp_path / "reports"))


def test_output_cannot_modify_input(dataset: Path) -> None:
    with pytest.raises(DatasetStructureError, match="пересек"):
        run_audit(AuditConfig(dataset, dataset / "reports"))


def test_existing_output_is_not_overwritten(dataset: Path, tmp_path: Path) -> None:
    output = tmp_path / "reports"
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(DatasetStructureError, match="непуст"):
        run_audit(AuditConfig(dataset, output))
    assert (output / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_duplicate_image_stems_are_ambiguous(dataset: Path, tmp_path: Path) -> None:
    (dataset / "train" / "a.png").write_bytes((dataset / "train" / "a.tif").read_bytes())
    result = run_audit(AuditConfig(dataset, tmp_path / "reports", examples=0))
    assert result.summary["image_files"] == 4
    assert any(issue.code == "duplicate_image_id" for issue in result.issues)
    assert result.summary["measurable_objects"] == 0


def test_duplicate_jsonl_records_do_not_inflate_geometry_statistics(
    dataset: Path,
    tmp_path: Path,
) -> None:
    path = dataset / "polygons.jsonl"
    first = path.read_text(encoding="utf-8").splitlines()[0]
    with path.open("a", encoding="utf-8") as stream:
        stream.write(first + "\n")
    result = run_audit(AuditConfig(dataset, tmp_path / "reports", examples=0))
    assert result.summary["objects"] == 6
    assert result.summary["measurable_objects"] == 0
    assert (
        next(row for row in result.images if row["image_id"] == "a")["annotation_presence"]
        == "invalid"
    )


def test_missing_wsi_fields_are_visible(dataset: Path, tmp_path: Path) -> None:
    (dataset / "wsi_meta.csv").write_text("source_wsi,age\n1,\n", encoding="utf-8")
    result = run_audit(AuditConfig(dataset, tmp_path / "reports", examples=0))
    assert any(issue.code == "missing_wsi_field" for issue in result.issues)


def test_damaged_tiff_dimensions_do_not_abort_audit(dataset: Path, tmp_path: Path) -> None:
    path = dataset / "train" / "b.tif"
    content = bytearray(path.read_bytes())
    content[130] = 248
    path.write_bytes(content)
    result = run_audit(AuditConfig(dataset, tmp_path / "reports", examples=0))
    assert result.summary["readable_images"] == 2
    assert any(issue.code == "corrupt_image" and issue.image_id == "b" for issue in result.issues)


def test_repeatable_statistics(dataset: Path, tmp_path: Path) -> None:
    first = run_audit(AuditConfig(dataset, tmp_path / "one", examples=0))
    second = run_audit(AuditConfig(dataset, tmp_path / "two", examples=0))
    assert first.summary == second.summary
    assert (first.output / "annotations.csv").read_bytes() == (
        second.output / "annotations.csv"
    ).read_bytes()


def test_empty_annotations_produce_complete_reports(dataset: Path, tmp_path: Path) -> None:
    (dataset / "polygons.jsonl").write_text("", encoding="utf-8")
    result = run_audit(AuditConfig(dataset, tmp_path / "reports", examples=0))
    assert result.summary["objects"] == 0
    assert result.summary["annotation_presence"] == {"missing": 3}
    assert json.loads((result.output / "geometry_statistics.json").read_text()) == []
    with (result.output / "annotations.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        assert "raw_json" in (reader.fieldnames or [])
        assert list(reader) == []
    assert (result.output / "audit.md").is_file()


def test_cli_runs_and_errors_are_readable(dataset: Path, tmp_path: Path) -> None:
    command = [
        sys.executable,
        "-m",
        "kidney_vessels",
        "audit",
        "--input",
        str(dataset),
        "--output",
        str(tmp_path / "cli"),
        "--examples",
        "1",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "cli" / "summary.json").is_file()
    command[command.index(str(dataset))] = str(tmp_path / "missing")
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8")
    assert completed.returncode == 2
    assert "Traceback" not in completed.stderr
    assert "Отсутствует обязательный путь" in completed.stderr


def test_cli_strict_returns_failure_after_writing_reports(dataset: Path, tmp_path: Path) -> None:
    with (dataset / "polygons.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("{broken}\n")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "kidney_vessels",
            "audit",
            "--input",
            str(dataset),
            "--output",
            str(tmp_path / "strict"),
            "--strict",
            "--examples",
            "0",
        ],
        capture_output=True,
    )
    assert completed.returncode == 1
    assert (tmp_path / "strict" / "issues.csv").is_file()
