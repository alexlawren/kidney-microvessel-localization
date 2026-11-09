"""Предварительная проверка формата локальных checkpoint практики без десериализации."""

from pathlib import Path, PurePosixPath
from zipfile import ZIP_STORED, BadZipFile, ZipFile

MAX_ARCHIVE_BYTES = 256 * 1024**2
MAX_PICKLE_BYTES = 1024**2
MAX_MEMBERS = 2048


def validate_checkpoint_archive(path: Path) -> None:
    """Допустить только ограниченный по размеру ZIP формата torch.save для state_dict."""
    if not 0 < path.stat().st_size <= MAX_ARCHIVE_BYTES:
        raise ValueError("Размер файла checkpoint вне допустимого диапазона")
    try:
        with ZipFile(path) as archive:
            entries = archive.infolist()
            if not 0 < len(entries) <= MAX_MEMBERS:
                raise ValueError("Недопустимое число записей в checkpoint")
            if sum(entry.file_size for entry in entries) > MAX_ARCHIVE_BYTES:
                raise ValueError("Суммарный размер записей checkpoint превышает предел")
            names = [PurePosixPath(entry.filename) for entry in entries]
            if any(
                len(name.parts) < 2
                or name.is_absolute()
                or ".." in name.parts
                or "\\" in entry.filename
                or entry.is_dir()
                or entry.compress_type != ZIP_STORED
                or entry.flag_bits & 1
                for name, entry in zip(names, entries, strict=True)
            ):
                raise ValueError("Недопустимая запись ZIP в checkpoint")
            if len(set(names)) != len(names) or len({name.parts[0] for name in names}) != 1:
                raise ValueError("В checkpoint ожидается один корневой каталог без дубликатов")
            if any(name.name == "constants.pkl" for name in names):
                raise ValueError("TorchScript checkpoint не поддерживается")
            metadata = [
                entry
                for name, entry in zip(names, entries, strict=True)
                if len(name.parts) == 2 and name.name == "data.pkl"
            ]
            if len(metadata) != 1 or not 0 < metadata[0].file_size <= MAX_PICKLE_BYTES:
                raise ValueError("Ожидается одна ограниченная по размеру запись data.pkl")
    except BadZipFile as exc:
        raise ValueError("Ожидается ZIP checkpoint, созданный torch.save") from exc
