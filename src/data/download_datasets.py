from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
import hashlib

import pandas as pd
from ucimlrepo import fetch_ucirepo


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

QSAR_URL = (
    "https://archive.ics.uci.edu/static/public/254/"
    "qsar%2Bbiodegradation.zip"
)

QSAR_FEATURES = [
    "SpMax_L",
    "J_Dz(e)",
    "nHM",
    "F01[N-N]",
    "F04[C-N]",
    "NssssC",
    "nCb-",
    "C%",
    "nCp",
    "nO",
    "F03[C-N]",
    "SdssC",
    "HyWi_B(m)",
    "LOC",
    "SM6_L",
    "F03[C-O]",
    "Me",
    "Mi",
    "nN-N",
    "nArNO2",
    "nCRX3",
    "SpPosA_B(p)",
    "nCIR",
    "B01[C-Br]",
    "B03[C-Cl]",
    "N-073",
    "SpMax_A",
    "Psi_i_1d",
    "B04[C-Br]",
    "SdO",
    "TI2_L",
    "nCrt",
    "C-026",
    "F02[C-N]",
    "nHDon",
    "SpMax_B(m)",
    "Psi_i_A",
    "nN",
    "SM6_B(m)",
    "nArCOOR",
    "nX",
]


def calculate_sha256(path: Path) -> str:
    """Calculate SHA-256 checksum of a file."""
    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def save_dataset(data: pd.DataFrame, filename: str) -> None:
    """Save dataset and print basic information."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RAW_DIR / filename
    data.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print(f"Shape: {data.shape}")
    print(f"SHA-256: {calculate_sha256(output_path)}")


def download_german_credit() -> None:
    """Download German Credit from UCI."""
    print("\nDownloading German Credit...")

    dataset = fetch_ucirepo(id=144)

    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError("Expected exactly one target column.")
        y = y.iloc[:, 0]

    data = X.copy()
    data["target"] = y

    save_dataset(data, "german_credit.csv")


def download_qsar() -> None:
    """Download QSAR biodegradation directly from the UCI archive."""
    print("\nDownloading QSAR biodegradation...")

    with urlopen(QSAR_URL, timeout=60) as response:
        archive_bytes = response.read()

    with ZipFile(BytesIO(archive_bytes)) as archive:
        csv_files = [
            name for name in archive.namelist()
            if Path(name).name.lower() == "biodeg.csv"
        ]

        if len(csv_files) != 1:
            raise ValueError(
                f"Expected one biodeg.csv file, found: {csv_files}"
            )

        with archive.open(csv_files[0]) as file:
            data = pd.read_csv(
                file,
                sep=";",
                header=None,
                skipinitialspace=True,
            )

    if data.shape[1] != 42:
        raise ValueError(
            f"Expected 42 columns (41 features + target), "
            f"got {data.shape[1]}"
        )

    data.columns = QSAR_FEATURES + ["target"]
    data["target"] = data["target"].astype(str).str.strip()

    save_dataset(data, "qsar_biodegradation.csv")


def main() -> None:
    download_german_credit()
    download_qsar()

    print("\nAll datasets downloaded successfully.")


if __name__ == "__main__":
    main()
