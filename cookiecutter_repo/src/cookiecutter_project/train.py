from pathlib import Path


import matplotlib.pyplot as plt
import torch
import typer

from cookiecutter_project.model import MyAwesomeModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_processed(processed_dir: Path = Path("data/processed")):
    train_images = torch.load(processed_dir / "train_images.pt")
    train_target = torch.load(processed_dir / "train_target.pt")
    test_images = torch.load(processed_dir / "test_images.pt")
    test_target = torch.load(processed_dir / "test_target.pt")
    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)
    return train_set, test_set

def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 10) -> None:
    print("Training day and night")
    print(f"{lr=}, {batch_size=}, {epochs=}")

    model = MyAwesomeModel().to(DEVICE)
    train_set, _ = load_processed()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(epochs):
        model.train()
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()

            statistics["train_loss"].append(loss.item())
            statistics["train_accuracy"].append((y_pred.argmax(dim=1) == target).float().mean().item())

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

    Path("models").mkdir(parents=True, exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), "models/model.pth")

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"]); axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"]); axs[1].set_title("Train accuracy")
    fig.savefig("reports/figures/training_statistics.png")

if __name__ == "__main__":
    typer.run(train)