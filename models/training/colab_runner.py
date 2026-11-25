# Google Colab: mount Google Drive, copy the data from it, and keep the checkpoints and the training history on it.
# The notebook's Setup cell calls setup(). To only train, without the notebook, run in a Colab cell:
#   !python models/training/colab_runner.py
# Upload these once to "My Drive/miniGPT/":
#   raw/ar.parquet, raw/en.parquet        (from datasets/raw/, used by the notebook stages 1-6)
#   splits/train.parquet, splits/val.parquet  (from datasets/splits/, used by the tokenizer and the training)
from pathlib import Path
import os
import shutil
import subprocess
import sys

from google.colab import drive

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DRIVE_DIR = Path("/content/drive/MyDrive/miniGPT")
LOCAL_ARTIFACTS_DIR = Path("/content/miniGPT_artifacts")  # token caches on Colab's fast local disk, rebuilt each session
FROM_DRIVE = {
    "raw": ["ar.parquet", "en.parquet"],
    "splits": ["train.parquet", "val.parquet"],  # the test split is not used
}


# Copy the data files from Drive into datasets/, and stop if one is missing on Drive.
def copy_data():
    for folder, names in FROM_DRIVE.items():
        (PROJECT_ROOT / "datasets" / folder).mkdir(parents=True, exist_ok=True)
        for name in names:
            source = DRIVE_DIR / folder / name
            if not source.exists():
                sys.exit(f"{source} is missing: upload datasets/{folder}/{name} to My Drive/miniGPT/{folder}/ first.")
            shutil.copy2(source, PROJECT_ROOT / "datasets" / folder / name)
            print(f"  copied {folder}/{name} ({source.stat().st_size / 1e6:.0f} MB)")


# Mount Drive in this kernel, copy the data, and send checkpoints and history to Drive (before training is imported).
def setup():
    print("\n### Colab setup")
    drive.mount("/content/drive")
    copy_data()
    os.environ["MINIGPT_ARTIFACTS_DIR"] = str(LOCAL_ARTIFACTS_DIR)
    os.environ["MINIGPT_CHECKPOINTS_DIR"] = str(DRIVE_DIR / "checkpoints")
    print(f"  checkpoints and history.json go to {DRIVE_DIR / 'checkpoints'}")


# Train without the notebook: set up, then start, or continue the checkpoint on Drive if there is one.
def main():
    setup()
    command = [sys.executable, str(PROJECT_ROOT / "models" / "training" / "run_training.py"), "--resume"]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
