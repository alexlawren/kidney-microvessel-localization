"""Производные RGB-копии для просмотра; исходные изображения не изменяются."""

from pathlib import Path

from PIL import Image


def load_rgb_preview(path: Path) -> Image.Image:
    """Получить независимую RGB-копию тайла без нормализации окраски и геометрии."""
    with Image.open(path) as image:
        image.load()
        return image.convert("RGB")
