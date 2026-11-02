"""Декодирование изображений для аудита целостности."""

import struct
import warnings
from pathlib import Path

from PIL import Image


class ImageReadError(ValueError):
    """Изображение не удалось безопасно декодировать."""


def inspect_image(path: Path) -> tuple[int, int, str]:
    """Полностью декодировать единственную страницу, вернуть размеры и режим."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                if getattr(image, "n_frames", 1) != 1:
                    raise ImageReadError(f"Ожидается одностраничный тайл: {path}")
                image.load()
                return image.width, image.height, image.mode
    except (
        OSError,
        ValueError,
        SyntaxError,
        TypeError,
        KeyError,
        IndexError,
        struct.error,
        UserWarning,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ImageReadError(f"Не удалось прочитать изображение {path}: {exc}") from exc
