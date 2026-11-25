# Train on Google Colab: copy the splits from Google Drive, keep the checkpoints on Drive, and resume if one is there.
# Upload datasets/splits/train.parquet and val.parquet to "My Drive/miniGPT/splits/" once, then in Colab:
#   !python models/training/colab_runner.py
from pathlib import Path
import os
import shutil
import subprocess
import sys

from google.colab import drive

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DRIVE_DIR = Path("/content/drive/MyDrive/miniGPT")
DRIVE_SPLITS_DIR = DRIVE_DIR / "splits"            # uploaded once by hand
DRIVE_CHECKPOINTS_DIR = DRIVE_DIR / "checkpoints"  # survives when the Colab session drops
LOCAL_ARTIFACTS_DIR = Path("/content/miniGPT_artifacts")  # token caches on Colab's fast local disk, rebuilt each session
SPLITS = ["train.parquet", "val.parquet"]  # the test split is not used in training


# Copy the splits the training reads from Drive into datasets/splits/, and stop if one is missing on Drive.
def copy_splits():
    splits_dir = PROJECT_ROOT / "datasets" / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLITS:
        source = DRIVE_SPLITS_DIR / name
        if not source.exists():
            sys.exit(f"{source} is missing: upload datasets/splits/{name} to My Drive/miniGPT/splits/ first.")
        shutil.copy2(source, splits_dir / name)
        print(f"copied {name} ({source.stat().st_size / 1e6:.0f} MB)")


# Mount Drive, copy the splits, point the checkpoints to Drive, then start or resume the training.
def main():
    drive.mount("/content/drive")
    copy_splits()
    os.environ["MINIGPT_ARTIFACTS_DIR"] = str(LOCAL_ARTIFACTS_DIR)
    os.environ["MINIGPT_CHECKPOINTS_DIR"] = str(DRIVE_CHECKPOINTS_DIR)

    command = [sys.executable, str(PROJECT_ROOT / "models" / "training" / "run_training.py")]
    if (DRIVE_CHECKPOINTS_DIR / "last_checkpoint.pt").exists():
        command.append("--resume")
    else:
        print("no checkpoint on Drive yet: starting a new training run")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
