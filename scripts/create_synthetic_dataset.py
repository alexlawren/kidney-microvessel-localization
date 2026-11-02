"""Создать воспроизводимый искусственный пример; это не медицинские изображения."""

import argparse
import csv
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw


def create_dataset(root: Path) -> None:
    """Сохранить три демонстрационных тайла в исходном формате HuBMAP."""
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"Каталог должен быть новым или пустым: {root}")
    (root / "train").mkdir(parents=True)
    rng = random.Random(20261102)
    for image_id in ("synthetic_1", "synthetic_2", "synthetic_3"):
        image = Image.new("RGB", (256, 256), (244, 217, 230))
        draw = ImageDraw.Draw(image)
        for _ in range(90):
            x, y = rng.randrange(256), rng.randrange(256)
            draw.ellipse((x, y, x + 12, y + 8), fill=(195, 145, 185), outline=(140, 85, 145))
        draw.text((8, 238), "SYNTHETIC - NOT CLINICAL DATA", fill=(60, 25, 70))
        image.save(root / "train" / f"{image_id}.tif")
        image.close()
    with (root / "tile_meta.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "source_wsi", "dataset", "i", "j"])
        writer.writerows(
            [
                ["synthetic_1", "1", "1", "0", "0"],
                ["synthetic_2", "1", "2", "256", "0"],
                ["synthetic_3", "6", "3", "0", "0"],
            ]
        )
    (root / "wsi_meta.csv").write_text(
        "source_wsi,age,sex,race,height,weight,bmi\n1,,,,,,\n", encoding="utf-8"
    )
    records = [
        {
            "id": "synthetic_1",
            "annotations": [
                {"type": "blood_vessel", "coordinates": [[[24, 24], [90, 30], [85, 66], [30, 72]]]},
                {
                    "type": "glomerulus",
                    "coordinates": [[[130, 28], [205, 28], [205, 100], [130, 100]]],
                },
                {"type": "unsure", "coordinates": [[[56, 132], [112, 140], [92, 190]]]},
            ],
        },
        {"id": "synthetic_2", "annotations": []},
    ]
    (root / "polygons.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    (root / "SYNTHETIC_DATASET.md").write_text(
        "# Искусственный тестовый набор\n\nЭто геометрические рисунки, не ткани человека.\n"
        "Seed: 20261102. Формат соответствует проверяемому контракту HuBMAP 2023.\n",
        encoding="utf-8",
    )


def main() -> None:
    """Запустить генератор из корня проекта."""
    parser = argparse.ArgumentParser(description="Создать искусственный HuBMAP-подобный пример")
    parser.add_argument("--output", type=Path, default=Path("data/synthetic/hubmap"))
    args = parser.parse_args()
    try:
        create_dataset(args.output)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Ошибка: {exc}\n")


if __name__ == "__main__":
    main()
