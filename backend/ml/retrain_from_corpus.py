"""
Offline retrain helper for PhysioTracker.

Usage:
  cd backend
  python -m ml.retrain_from_corpus

The model now auto-loads real labeled windows from backend/data/imu_corpus.jsonl.
This script simply forces a fresh training pass and prints training info.
"""
from ml.model import PhysioMLModel


def main():
    model = PhysioMLModel()
    info = model.get_info()
    print("Retrain complete")
    print(f"trained={info.get('trained')}")
    print(f"accuracy={info.get('accuracy')}")
    print(f"samples_trained={info.get('samples_trained')}")
    print(f"real_windows={info.get('real_windows')}")
    print(f"model={info.get('model')}")


if __name__ == "__main__":
    main()
