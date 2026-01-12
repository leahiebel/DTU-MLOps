import pytorch_lightning as pl
from torch.utils.data import DataLoader

from cookiecutter_project.data import corrupt_mnist  # or wherever your function is
from lightning import MyAwesomeModel


def main():
    # Q3: dataset -> dataloader
    train_set, test_set = corrupt_mnist()

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0)

    model = MyAwesomeModel()

    # Q4: trainer flags
    trainer = pl.Trainer(
        default_root_dir="lightning_runs",  # <- where logs/checkpoints go
        max_epochs=10,                      # <- not 1000
        # max_steps=500,                    # <- optional alternative
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
        limit_train_batches=0.2,  # use a fraction of training set for faster epoch
    )

    # Q5: train
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()