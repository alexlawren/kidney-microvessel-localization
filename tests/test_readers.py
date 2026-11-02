"""Контракты чтения CSV, исходного JSONL и изображений."""

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from PIL import Image

from kidney_vessels.data.hubmap import DatasetStructureError, read_annotations, read_metadata
from kidney_vessels.data.images import ImageReadError, inspect_image


def test_actual_annotation_contract_and_raw_fields(dataset: Path) -> None:
    result = read_annotations(dataset / "polygons.jsonl", {"a": (32, 32)})
    assert len(result.records) == 2
    record = result.records[0]
    assert record.image_id == "a"
    assert len(record.annotations) == 3
    assert record.annotations[0].category is not None
    assert record.annotations[0].category.name == "blood_vessel"
    assert record.annotations[0].geometry is not None
    assert record.annotations[0].geometry.area == 100
    assert json.loads(record.annotations[0].raw_json)["type"] == "blood_vessel"
    assert result.records[1].annotations == ()


@pytest.mark.parametrize(
    "content", ["{bad}\n", "[]\n", "{}\n", '{"id":"a"}\n', '{"id":"a","annotations":null}\n']
)
def test_invalid_json_and_structure_are_diagnosed(
    set_annotations: Callable[[str], Path], content: str
) -> None:
    result = read_annotations(set_annotations(content), {})
    assert any(issue.severity == "error" for issue in result.issues)
    assert result.records[0].raw_line.strip() == content.strip()
    assert not result.records[0].valid


def test_nonstandard_json_numbers_are_rejected(set_annotations: Callable[[str], Path]) -> None:
    result = read_annotations(set_annotations('{"id":"a","annotations":[NaN]}'), {})
    assert result.issues[0].code == "invalid_json"


def test_overflowing_coordinates_do_not_discard_other_objects(
    set_annotations: Callable[[str], Path],
) -> None:
    payload = {
        "id": "a",
        "annotations": [
            {"type": "blood_vessel", "coordinates": [[[10**400, 1], [3, 1], [3, 3]]]},
            {"type": "blood_vessel", "coordinates": [[[1, 1], [3, 1], [3, 3]]]},
        ],
    }
    result = read_annotations(set_annotations(json.dumps(payload)), {"a": (32, 32)})
    assert len(result.records[0].annotations) == 2
    assert result.records[0].annotations[0].errors
    assert result.records[0].annotations[1].geometry is not None


def test_overflowing_json_float_is_diagnosed(set_annotations: Callable[[str], Path]) -> None:
    result = read_annotations(
        set_annotations('{"id":"a","annotations":[{"coordinates":[[[1e400,0]]]}]}'), {}
    )
    assert result.issues[0].code == "invalid_json"


def test_empty_file_is_valid_but_does_not_imply_negative(
    set_annotations: Callable[[str], Path],
) -> None:
    result = read_annotations(set_annotations("\n"), {})
    assert result.records == ()


def test_unknown_label_is_preserved(set_annotations: Callable[[str], Path]) -> None:
    raw = {
        "id": "a",
        "annotations": [
            {"type": "novel", "extra": "retain", "coordinates": [[[1, 1], [4, 1], [4, 4]]]}
        ],
    }
    result = read_annotations(set_annotations(json.dumps(raw)), {"a": (32, 32)})
    annotation = result.records[0].annotations[0]
    assert annotation.category is not None
    assert annotation.category.name == "novel"
    assert json.loads(annotation.raw_json)["extra"] == "retain"
    assert "unknown_category" in annotation.warnings


def test_invalid_object_does_not_hide_following_object(
    set_annotations: Callable[[str], Path],
) -> None:
    result = read_annotations(
        set_annotations(
            json.dumps(
                {
                    "id": "a",
                    "annotations": [
                        None,
                        {"type": "blood_vessel", "coordinates": [[[1, 1], [4, 1], [4, 4]]]},
                    ],
                }
            )
        ),
        {"a": (32, 32)},
    )
    assert len(result.records[0].annotations) == 2
    assert result.records[0].annotations[0].errors
    assert result.records[0].annotations[1].geometry is not None


def test_duplicate_annotation_records_are_not_overwritten(
    set_annotations: Callable[[str], Path],
) -> None:
    line = '{"id":"a","annotations":[]}'
    result = read_annotations(set_annotations(line + "\n" + line), {})
    assert len(result.records) == 2
    assert any(issue.code == "duplicate_annotation_record" for issue in result.issues)


def test_metadata_preserves_raw_strings_and_reports_duplicates(dataset: Path) -> None:
    path = dataset / "tile_meta.csv"
    with path.open("a", encoding="utf-8") as stream:
        stream.write("a,2,2,0,0\n")
    table = read_metadata(path, "id", {"id", "source_wsi", "dataset", "i", "j"})
    assert table.by_id["a"]["source_wsi"] == "1"
    assert len(table.rows) == 4
    assert any(issue.code == "duplicate_metadata_id" for issue in table.issues)


def test_wrong_metadata_schema_fails(dataset: Path) -> None:
    path = dataset / "tile_meta.csv"
    path.write_text("image,source\na,1\n", encoding="utf-8")
    with pytest.raises(DatasetStructureError, match="id"):
        read_metadata(path, "id", {"id", "source_wsi"})


def test_image_is_fully_decoded_and_corrupt_file_is_rejected(dataset: Path) -> None:
    assert inspect_image(dataset / "train" / "a.tif") == (32, 32, "RGB")
    path = dataset / "train" / "broken.tif"
    path.write_bytes(b"not an image")
    with pytest.raises(ImageReadError):
        inspect_image(path)


def test_multipage_image_is_rejected(dataset: Path) -> None:
    path = dataset / "train" / "pages.tif"
    first = Image.new("RGB", (8, 8))
    second = Image.new("RGB", (8, 8))
    first.save(path, save_all=True, append_images=[second])
    first.close()
    second.close()
    with pytest.raises(ImageReadError, match="одностраничный"):
        inspect_image(path)
