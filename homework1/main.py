"""Полносвязная сеть для MNIST: NumPy, ReLU, Softmax и cross-entropy."""

import argparse
import json
from pathlib import Path
import shutil
import urllib.error
import urllib.request

import numpy as np


ROOT = Path(__file__).resolve().parent
MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"


class Net:
    def __init__(self, input_size=784, hidden_size=128, output_size=10,
                 lr=0.1, seed=42):
        self.lr = lr
        self.rng = np.random.default_rng(seed)
        # Инициализация He помогает сохранять масштаб активаций ReLU.
        self.weight_input_hidden = (
            self.rng.standard_normal((input_size, hidden_size))
            * np.sqrt(2.0 / input_size)
        ).astype(np.float32)
        self.weight_hidden_output = (
            self.rng.standard_normal((hidden_size, output_size))
            * np.sqrt(2.0 / hidden_size)
        ).astype(np.float32)
        self.bias_hidden = np.zeros(hidden_size, dtype=np.float32)
        self.bias_output = np.zeros(output_size, dtype=np.float32)

    @staticmethod
    def relu(layer):
        return np.maximum(0, layer)

    @staticmethod
    def softmax(logits):
        # Сдвиг не меняет вероятности, но предотвращает переполнение exp.
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exponents = np.exp(shifted)
        return exponents / np.sum(exponents, axis=1, keepdims=True)

    def forward_propagation(self, X):
        # Каждая строка — один пример; один снимок 28x28 тоже допустим.
        self.input = np.asarray(X).reshape(-1, self.weight_input_hidden.shape[0])
        self.hidden = self.relu(
            self.input @ self.weight_input_hidden + self.bias_hidden
        )
        self.logits = self.hidden @ self.weight_hidden_output + self.bias_output
        self.output = self.softmax(self.logits)
        return self.output

    def one_hot_encoding(self, y):
        labels = np.asarray(y, dtype=np.int64).reshape(-1)
        one_hot = np.zeros((len(labels), len(self.bias_output)), dtype=np.float32)
        one_hot[np.arange(len(labels)), labels] = 1
        return one_hot

    @staticmethod
    def cross_entropy(logits, y):
        """Средняя -log(p верного класса), вычисленная устойчиво из logits."""
        labels = np.asarray(y, dtype=np.int64).reshape(-1)
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        log_probabilities = shifted - np.log(
            np.sum(np.exp(shifted), axis=1, keepdims=True)
        )
        return float(-np.mean(log_probabilities[np.arange(len(labels)), labels]))

    def gradients(self, y):
        # Для Softmax + cross-entropy производная по logits: (p - y) / N.
        delta_output = (self.output - self.one_hot_encoding(y)) / len(self.input)
        # Производная ReLU равна 1 при положительной активации и 0 иначе.
        delta_hidden = (
            delta_output @ self.weight_hidden_output.T
        ) * (self.hidden > 0)
        return {
            "weight_input_hidden": self.input.T @ delta_hidden,
            "bias_hidden": delta_hidden.sum(axis=0),
            "weight_hidden_output": self.hidden.T @ delta_output,
            "bias_output": delta_output.sum(axis=0),
        }

    def back_propagation(self, y):
        # Сначала считаем ВСЕ градиенты со старыми весами, затем обновляем веса.
        for name, gradient in self.gradients(y).items():
            parameter = getattr(self, name)
            parameter -= self.lr * gradient

    def predict(self, X):
        return np.argmax(self.forward_propagation(X), axis=1)

    def evaluate(self, X, y, batch_size=1024):
        total_loss, correct = 0.0, 0
        for start in range(0, len(X), batch_size):
            labels = y[start:start + batch_size]
            probabilities = self.forward_propagation(X[start:start + batch_size])
            total_loss += self.cross_entropy(self.logits, labels) * len(labels)
            correct += np.count_nonzero(probabilities.argmax(axis=1) == labels)
        return total_loss / len(X), correct / len(X)

    def fit(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=128):
        history = {key: [] for key in (
            "train_loss", "val_loss", "train_accuracy", "val_accuracy"
        )}
        for epoch in range(1, epochs + 1):
            indices = self.rng.permutation(len(X_train))
            for start in range(0, len(indices), batch_size):
                batch = indices[start:start + batch_size]
                self.forward_propagation(X_train[batch])
                self.back_propagation(y_train[batch])

            # Обе выборки оцениваются с одинаковыми весами в конце эпохи.
            train_loss, train_accuracy = self.evaluate(X_train, y_train)
            val_loss, val_accuracy = self.evaluate(X_val, y_val)
            for key, value in zip(history, (
                train_loss, val_loss, train_accuracy, val_accuracy
            )):
                history[key].append(value)
            print(
                f"Epoch {epoch:2d}/{epochs}: "
                f"loss={train_loss:.4f}, accuracy={train_accuracy:.2%} | "
                f"val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.2%}",
                flush=True,
            )
        return history


def load_mnist(data_dir):
    """Тот же архив MNIST, что использует Keras, без зависимости от TensorFlow."""
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "mnist.npz"
    if not archive.exists():
        print("Downloading MNIST...", flush=True)
        temporary = archive.with_suffix(".npz.part")
        try:
            with urllib.request.urlopen(MNIST_URL, timeout=60) as response:
                with temporary.open("wb") as destination:
                    shutil.copyfileobj(response, destination)
            temporary.replace(archive)
        except (OSError, urllib.error.URLError) as error:
            raise RuntimeError(
                f"Не удалось скачать MNIST. Проверьте интернет или сохраните "
                f"{MNIST_URL} в {archive}."
            ) from error
        finally:
            temporary.unlink(missing_ok=True)
    with np.load(archive, allow_pickle=False) as data:
        X_train = data["x_train"].reshape(-1, 784).astype(np.float32) / 255.0
        X_test = data["x_test"].reshape(-1, 784).astype(np.float32) / 255.0
        return (X_train, data["y_train"]), (X_test, data["y_test"])


def plot_history(history, output_dir, show=True):
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for ax, metric, title, ylabel in (
        (axes[0], "loss", "Cross-entropy loss", "Loss"),
        (axes[1], "accuracy", "Classification accuracy", "Accuracy"),
    ):
        ax.plot(epochs, history[f"train_{metric}"], marker="o", label="Train")
        ax.plot(epochs, history[f"val_{metric}"], marker="o", label="Validation")
        ax.set(title=title, xlabel="Epoch", ylabel=ylabel, xticks=epochs)
        ax.grid(alpha=0.25)
        ax.legend()
    axes[1].set_ylim(0, 1)
    fig.suptitle("MNIST · 784 → 128 (ReLU) → 10 (Softmax)")
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / "training_curves.png"
    fig.savefig(figure_path, dpi=160)
    print(f"Graphs saved: {figure_path}", flush=True)
    if show:
        plt.show()
    plt.close(fig)


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Значение должно быть положительным.")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=positive_int, default=10)
    parser.add_argument("--batch-size", type=positive_int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--no-show", action="store_true",
                        help="Сохранить графики без открытия окна.")
    args = parser.parse_args()
    if not np.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr должен быть конечным положительным числом.")

    (X, y), (X_test, y_test) = load_mnist(args.data_dir)
    # 55 000 примеров для обучения, 5 000 для валидации.
    # Тестовая выборка (10 000) используется только для итоговой оценки.
    indices = np.random.default_rng(args.seed).permutation(len(X))
    val_indices, train_indices = indices[:5000], indices[5000:]
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    print(f"Train: {len(X_train)}, validation: {len(X_val)}, test: {len(X_test)}",
          flush=True)
    net = Net(lr=args.lr, seed=args.seed)
    history = net.fit(X_train, y_train, X_val, y_val,
                      epochs=args.epochs, batch_size=args.batch_size)
    test_loss, test_accuracy = net.evaluate(X_test, y_test)
    print(f"Test: loss={test_loss:.4f}, accuracy={test_accuracy:.2%}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "config": {"epochs": args.epochs, "batch_size": args.batch_size,
                   "lr": args.lr, "seed": args.seed, "architecture": [784, 128, 10]},
        "history": history,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    plot_history(history, args.output_dir, show=not args.no_show)


if __name__ == "__main__":
    main()
