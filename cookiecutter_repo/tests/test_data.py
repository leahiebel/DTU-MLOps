from cookiecutter_project.data import corrupt_mnist, preprocess_data
from pathlib import Path
import torch

def test_data():
    train, test = corrupt_mnist()
    assert len(train) == 30000, "Train dataset should have 30000 samples" 
    assert len(test) == 5000, "Test dataset should have 5000 samples" #added an assertion for test set size to know both sizes are correct
    for dataset in [train, test]:
        for x, y in dataset:
            assert x.shape == (1, 28, 28)
            assert y in range(10)
    train_targets = torch.unique(train.tensors[1])
    assert (train_targets == torch.arange(0,10)).all()
    test_targets = torch.unique(test.tensors[1])
    assert (test_targets == torch.arange(0,10)).all()

def test_preprocess_data_creates_processed_files(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    processed_dir.mkdir()

    # fake raw data in the exact filenames preprocess_data expects
    for i in range(6):
        torch.save(torch.ones(2, 28, 28) * i, raw_dir / f"train_images_{i}.pt")
        torch.save(torch.tensor([i, i]), raw_dir / f"train_target_{i}.pt")

    torch.save(torch.ones(2, 28, 28) * 10, raw_dir / "test_images.pt")
    torch.save(torch.tensor([1, 2]), raw_dir / "test_target.pt")

    preprocess_data(str(raw_dir), str(processed_dir))

    # outputs exist
    for fname in ["train_images.pt", "train_target.pt", "test_images.pt", "test_target.pt"]:
        assert (processed_dir / fname).exists()

        # basic sanity checks
    train_images = torch.load(processed_dir / "train_images.pt")
    test_images = torch.load(processed_dir / "test_images.pt")

    assert train_images.shape[1:] == (1, 28, 28)  # channel dim added
    assert test_images.shape[1:] == (1, 28, 28)

    assert train_images.dtype == torch.float32
    assert abs(train_images.mean().item()) < 1e-5
