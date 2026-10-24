import argparse
from pathlib import Path

SPLITS = ("train", "dev")


# Size in bytes of {directory}/{lang}/{split}.txt, or None if the file is missing.
def byte_count(directory: Path, lang: str, split: str) -> int | None:
    path = directory / lang / f"{split}.txt"
    return path.stat().st_size if path.is_file() else None


def format_count(count: int | None) -> str:
    return f"{count:,}" if count is not None else "-"


def format_kept(raw: int | None, clean: int | None) -> str:
    return f"{clean / raw:.1%}" if raw and clean is not None else "-"


def main() -> None:
    parser = argparse.ArgumentParser(description="CS489 A1 Part 1.1: raw and cleaned byte counts per language.")
    parser.add_argument("--raw-input", required=True, type=Path, help="e.g. data/raw/")
    parser.add_argument("--clean-input", required=True, type=Path, help="e.g. artifacts/part1/cleaned/")
    args = parser.parse_args()

    languages = sorted(p.name for p in args.raw_input.iterdir() if p.is_dir())
    header = f"{'lang':<6}" + "".join(
        f"{label:>14}" for split in SPLITS for label in (f"raw {split}", f"clean {split}", "kept")
    )
    print(header)
    for lang in languages:
        row = f"{lang:<6}"
        for split in SPLITS:
            raw = byte_count(args.raw_input, lang, split)
            clean = byte_count(args.clean_input, lang, split)
            row += f"{format_count(raw):>14}{format_count(clean):>14}{format_kept(raw, clean):>14}"
        print(row)


if __name__ == "__main__":
    main()
