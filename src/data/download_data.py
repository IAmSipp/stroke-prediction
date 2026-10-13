from argparse import ArgumentParser
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def download_dataset(dataset: str, force: bool = False) -> Path:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing_files = [path for path in RAW_DATA_DIR.iterdir() if path.name != ".gitkeep"]

    if existing_files and not force:
        print(f"Raw data already exists in {RAW_DATA_DIR}; skipping download.")
        return RAW_DATA_DIR

    import kaggle

    kaggle.api.dataset_download_files(
        dataset=dataset,
        path=str(RAW_DATA_DIR),
        unzip=True,
        quiet=False,
        force=force,
    )
    print(f"Dataset is ready in {RAW_DATA_DIR}")
    return RAW_DATA_DIR


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        help="Kaggle dataset identifier in owner/dataset-name format",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download again even when data/raw already contains files",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_dataset(args.dataset, force=args.force)
