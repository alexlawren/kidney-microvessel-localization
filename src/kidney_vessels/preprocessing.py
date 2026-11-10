"""Производные RGB-копии для просмотра; исходные изображения не изменяются."""

from pathlib import Path

from PIL import Image, ImageDraw

from kidney_vessels.domain import PolygonGeometry
from kidney_vessels.evaluation import BinaryMask


def load_rgb_preview(path: Path) -> Image.Image:
    """Получить независимую RGB-копию тайла без нормализации окраски и геометрии."""
    with Image.open(path) as image:
        image.load()
        return image.convert("RGB")


def rasterize_polygon(geometry: PolygonGeometry, size: tuple[int, int]) -> Image.Image:
    """Растеризовать компоненты одного объекта в маску принадлежности 0/255."""
    if not geometry.fits_tile(*size):
        raise ValueError("Геометрия выходит за границы растровой маски")
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for part in geometry.parts:
        draw.polygon([(point.x, point.y) for point in part], fill=255)
    return mask


def load_binary_mask(path: Path) -> BinaryMask:
    """Прочитать одноканальную маску 0/255 для метрик без неявного порога."""
    with Image.open(path) as image:
        if image.mode not in {"1", "L"} or getattr(image, "n_frames", 1) != 1:
            raise ValueError("Ожидается одностраничная одноканальная бинарная маска")
        with image.convert("L") as gray:
            pixels = gray.tobytes()
        if not set(pixels) <= {0, 255}:
            raise ValueError("Маска файла должна содержать только 0 и 255")
        return BinaryMask(image.width, image.height, pixels.replace(b"\xff", b"\x01"))
