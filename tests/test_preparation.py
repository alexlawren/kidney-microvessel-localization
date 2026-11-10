"""Геометрия целей, ручная коррекция и интеграционный экспорт HuBMAP."""

import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pytest
from PIL import Image

from kidney_vessels.annotations import validate_geometry
from kidney_vessels.application.prepare import PreparationConfig, run_preparation
from kidney_vessels.data.hubmap import read_annotations
from kidney_vessels.domain import GeometricAnnotation, HistologyImage, ImageOrigin, SourceCategory
from kidney_vessels.evaluation import MaskPair, evaluate_segmentation
from kidney_vessels.preparation import (
    GeometryPolicy,
    PolygonCorrection,
    annotation_sha256,
    prepare_annotation,
)
from kidney_vessels.preprocessing import load_binary_mask, rasterize_polygon

RECTANGLE = [[[1, 1], [4, 1], [4, 3], [1, 3]]]
INVALID = [[[1, 1], [4, 3], [1, 3], [4, 1]]]


def test_rasterization_and_bbox_preserve_source_coordinates() -> None:
    geometry = validate_geometry(RECTANGLE, 8, 8).geometry
    assert geometry is not None
    assert geometry.bbox == (1, 1, 4, 3)
    with rasterize_polygon(geometry, (8, 8)) as mask:
        assert mask.mode == "L"
        assert mask.getpixel((1, 1)) == mask.getpixel((4, 3)) == 255
        assert mask.getpixel((0, 0)) == mask.getpixel((5, 3)) == 0
        assert mask.histogram()[255] == 12


def test_skip_error_and_reviewed_correction_keep_original() -> None:
    tile = HistologyImage("a", Path("a.tif"), 8, 8, ImageOrigin("synthetic", "1"))
    raw = json.dumps({"type": "blood_vessel", "coordinates": INVALID})
    source = GeometricAnnotation(
        "line1:object0", "a", SourceCategory("blood_vessel"), None, raw, ("invalid_topology",)
    )
    skipped = prepare_annotation(source, tile, GeometryPolicy.SKIP)
    assert skipped.geometry is None and skipped.disposition == "invalid_geometry"
    with pytest.raises(ValueError, match="line1:object0"):
        prepare_annotation(source, tile, GeometryPolicy.ERROR)
    correction = PolygonCorrection(
        "line1:object0",
        "a",
        sha256(raw.encode()).hexdigest(),
        RECTANGLE,
        "Проверенный вручную контур",
    )
    corrected = prepare_annotation(source, tile, GeometryPolicy.ERROR, correction=correction)
    assert corrected.source is source and corrected.source.raw_json == raw
    assert corrected.geometry is not None and corrected.geometry.bbox == (1, 1, 4, 3)
    assert corrected.correction == correction
    stale = PolygonCorrection("line1:object0", "a", "0" * 64, RECTANGLE, "Проверка хеша")
    with pytest.raises(ValueError, match="SHA-256"):
        prepare_annotation(source, tile, GeometryPolicy.SKIP, correction=stale)


def test_prepare_cli_exports_targets_without_inventing_background(
    dataset: Path, tmp_path: Path
) -> None:
    source = dataset / "polygons.jsonl"
    original = source.read_bytes()
    output = tmp_path / "prepared"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "kidney_vessels",
            "prepare",
            "--input",
            str(dataset),
            "--output",
            str(output),
            "--weights",
            "1",
            "1",
            "0",
        ],
        capture_output=True,
        encoding="utf-8",
    )
    assert completed.returncode == 0, completed.stderr
    data = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    rows = {row["image_id"]: row for row in data["images"]}
    assert data["patient_independence"] == "not_verified"
    assert rows["a"]["target_coverage"] == "positive_only"
    vessel, unsure, glomerulus = rows["a"]["objects"]
    assert vessel["category"] == "blood_vessel" and vessel["bbox_xyxy"] == [2, 2, 12, 12]
    with Image.open(output / vessel["mask"]) as mask:
        assert mask.size == (32, 32) and mask.getpixel((2, 2)) == 255
    positive = load_binary_mask(output / vessel["mask"])
    scores = evaluate_segmentation([MaskPair(positive, positive, positive)])
    assert scores.micro.dice == 1 and scores.counts.evaluated_pixels == 121
    assert unsure["mask"] is None and unsure["bbox_xyxy"] is None
    assert glomerulus["category"] == "glomerulus"
    assert rows["b"]["annotation_completeness"] == "sparse"
    assert rows["c"]["annotation_completeness"] == "unannotated"
    for key in ("b", "c"):
        assert rows[key]["target_coverage"] == "unknown" and rows[key]["objects"] == []
    assert source.read_bytes() == original


def test_correction_manifest_and_atomic_failure(dataset: Path, tmp_path: Path) -> None:
    annotations = dataset / "polygons.jsonl"
    with annotations.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {"id": "c", "annotations": [{"type": "blood_vessel", "coordinates": INVALID}]}
            )
            + "\n"
        )
    table = dataset / "tile_meta.csv"
    table.write_text(
        table.read_text(encoding="utf-8").replace("c,6,3,", "c,6,2,"), encoding="utf-8"
    )
    source = read_annotations(annotations, {"c": (32, 32)}, skip_image_ids={"a", "b"})
    polygon = source.records[0].annotations[0]
    correction = {
        "annotation_id": polygon.annotation_id,
        "image_id": "c",
        "source_sha256": "0" * 64,
        "coordinates": RECTANGLE,
        "reason": "Экспертный контур",
    }
    edits = tmp_path / "corrections.json"
    edits.write_text(json.dumps([correction]), encoding="utf-8")
    output = tmp_path / "prepared"
    config = PreparationConfig(
        dataset, output, weights=(1, 1, 0), invalid_policy=GeometryPolicy.ERROR, corrections=edits
    )
    with pytest.raises(ValueError, match="SHA-256"):
        run_preparation(config)
    assert not output.exists()
    assert not list(tmp_path.glob(".prepared-*"))
    correction["source_sha256"] = annotation_sha256(polygon)
    edits.write_text(json.dumps([correction]), encoding="utf-8")
    manifest = json.loads(run_preparation(config).read_text(encoding="utf-8"))
    row = next(row for row in manifest["images"] if row["image_id"] == "c")
    assert row["annotation_presence"] == "invalid"
    assert row["objects"][0]["correction"] == correction
    assert row["objects"][0]["source_json"] == polygon.raw_json
    assert row["objects"][0]["bbox_xyxy"] == [1, 1, 4, 3]


def test_dataset_three_record_cannot_become_a_target(dataset: Path, tmp_path: Path) -> None:
    with (dataset / "polygons.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {"id": "c", "annotations": [{"type": "blood_vessel", "coordinates": RECTANGLE}]}
            )
            + "\n"
        )
    result = run_preparation(PreparationConfig(dataset, tmp_path / "prepared", weights=(1, 1, 0)))
    manifest = json.loads(result.read_text(encoding="utf-8"))
    tile = next(row for row in manifest["images"] if row["image_id"] == "c")
    assert tile["target_coverage"] == "unknown"
    assert tile["objects"][0]["disposition"] == "unannotated"
    assert tile["objects"][0]["mask"] is None


def test_gray_mask_requires_explicit_threshold(tmp_path: Path) -> None:
    path = tmp_path / "gray.png"
    Image.new("L", (1, 1), 128).save(path)
    with pytest.raises(ValueError, match="0 и 255"):
        load_binary_mask(path)


def test_wsi_cohort_is_selected_before_split_and_export_limit(
    dataset: Path, tmp_path: Path
) -> None:
    config = PreparationConfig(
        dataset,
        tmp_path / "prepared",
        weights=(1, 0, 0),
        source_wsi=frozenset({"1"}),
        limit_per_wsi=1,
    )
    manifest = json.loads(run_preparation(config).read_text(encoding="utf-8"))
    assert [(row["image_id"], row["exported"]) for row in manifest["assignments"]] == [
        ("a", True),
        ("b", False),
    ]
    assert manifest["split_coverage"]["train"] == {
        "exported_tiles": 1,
        "exported_tiles_with_targets": 1,
        "wsi_with_exported_targets": ["1"],
    }


def test_orphan_annotation_is_reported_even_with_export_limit(
    dataset: Path, tmp_path: Path
) -> None:
    record = json.dumps({"id": "missing", "annotations": []})
    with (dataset / "polygons.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(record + "\n")
    result = run_preparation(
        PreparationConfig(dataset, tmp_path / "prepared", weights=(1, 1, 0), limit_per_wsi=1)
    )
    manifest = json.loads(result.read_text(encoding="utf-8"))
    assert manifest["unlinked_records"][0]["raw_line"].strip() == record
    assert any(issue["code"] == "orphan_annotation" for issue in manifest["issues"])
    with pytest.raises(ValueError, match="без соответствующего train-тайла"):
        run_preparation(
            PreparationConfig(
                dataset,
                tmp_path / "strict",
                weights=(1, 1, 0),
                limit_per_wsi=1,
                invalid_policy=GeometryPolicy.ERROR,
            )
        )


def test_prepared_geometry_is_checked_against_current_tile_dimensions() -> None:
    geometry = validate_geometry(RECTANGLE, 8, 8).geometry
    assert geometry is not None
    tile = HistologyImage("a", Path("a.tif"), 3, 3, ImageOrigin("synthetic", "1"))
    annotation = GeometricAnnotation(
        "line1:object0", "a", SourceCategory("blood_vessel"), geometry, "{}"
    )
    with pytest.raises(ValueError, match="Границы аннотации"):
        prepare_annotation(annotation, tile, GeometryPolicy.SKIP)
