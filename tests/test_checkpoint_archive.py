"""Проверка предварительного отбора checkpoint без импорта PyTorch."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import pytest

from kidney_vessels.data.checkpoint_archive import validate_checkpoint_archive


def test_eager_zip_layout_is_accepted_for_further_validation(tmp_path: Path) -> None:
    path = tmp_path / "weights.pth"
    with ZipFile(path, "w", compression=ZIP_STORED) as archive:
        archive.writestr("weights/data.pkl", b"metadata")
        archive.writestr("weights/data/0", b"tensor")
    validate_checkpoint_archive(path)


@pytest.mark.parametrize(
    "name,payload,compression",
    [
        ("weights/constants.pkl", b"script", ZIP_STORED),
        ("other/data.pkl", b"duplicate", ZIP_STORED),
        ("weights/../other", b"unexpected", ZIP_STORED),
        ("weights/data/0", b"compressed", ZIP_DEFLATED),
    ],
    ids=["torchscript", "duplicate-pickle", "traversal", "compressed"],
)
def test_unexpected_checkpoint_layout_is_rejected(
    tmp_path: Path,
    name: str,
    payload: bytes,
    compression: int,
) -> None:
    path = tmp_path / "weights.pth"
    with ZipFile(path, "w", compression=compression) as archive:
        archive.writestr("weights/data.pkl", b"metadata")
        archive.writestr(name, payload)
    with pytest.raises(ValueError):
        validate_checkpoint_archive(path)


def test_oversized_checkpoint_metadata_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "weights.pth"
    with ZipFile(path, "w") as archive:
        archive.writestr("weights/data.pkl", b"x" * (1024**2 + 1))
    with pytest.raises(ValueError, match="data.pkl"):
        validate_checkpoint_archive(path)
