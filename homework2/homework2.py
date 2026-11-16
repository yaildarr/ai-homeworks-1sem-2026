from pathlib import Path
import re

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def tokenize(text):
    return re.findall(r"[а-яё]+(?:-[а-яё]+)*", text.lower())


class RNN(nn.Module):
    def __init__(self, words, hidden_size=15, context_size=25):
        super().__init__()
        self.vocab = sorted(set(words))
        self.word_to_id = {word: i for i, word in enumerate(self.vocab)}
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(len(self.vocab), 32)
        self.hidden_layer = nn.Linear(32 + hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, len(self.vocab))
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, inputs):
        # inputs: строки — примеры, столбцы — последовательные слова.
        vectors = self.embedding(inputs)
        hidden = vectors.new_zeros(inputs.size(0), self.hidden_size)
        for word_vector in vectors.unbind(dim=1):
            combined = torch.cat((word_vector, hidden), dim=1)
            hidden = torch.tanh(self.hidden_layer(combined))
        return self.output(hidden)

    def fit(self, words, epochs=20):
        if len(words) <= self.context_size:
            raise ValueError(f"Для обучения нужно хотя бы {self.context_size + 1} слов.")
        ids = torch.tensor([self.word_to_id[word] for word in words])
        windows = ids.unfold(0, self.context_size + 1, 1)
        dataset = TensorDataset(windows[:, :-1], windows[:, -1])
        batches = DataLoader(dataset, batch_size=64, shuffle=True)
        self.train()
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            for inputs, targets in batches:
                self.optimizer.zero_grad()
                scores = self(inputs)
                loss = self.criterion(scores, targets)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * targets.size(0)
                correct += (scores.argmax(dim=1) == targets).sum().item()
            print(f"Эпоха {epoch + 1}/{epochs} | Потери: {total_loss / len(dataset):.3f}"
                  f" | Точность на обучении: {correct / len(dataset):.1%}")

    @torch.no_grad()
    def predict(self, text):
        self.eval()
        words = tokenize(text)[-self.context_size:]
        if not words:
            raise ValueError("Введите хотя бы одно русское слово.")
        for word in words:
            if word not in self.word_to_id:
                raise ValueError(f"Слова «{word}» нет в словаре обучения.")
        inputs = torch.tensor([[self.word_to_id[word] for word in words]])
        next_id = self(inputs).argmax(dim=1).item()
        return self.vocab[next_id]


if __name__ == "__main__":
    path = Path(__file__).with_name("saltan.txt")
    if not path.is_file():
        raise SystemExit("Положите saltan.txt рядом с homework2.py.")
    words = tokenize(path.read_text(encoding="cp1251"))
    if len(words) < 26:
        raise SystemExit("В saltan.txt должно быть хотя бы 26 русских слов.")
    torch.manual_seed(42)
    model = RNN(words)
    model.fit(words)
    phrase = " ".join(words[-model.context_size:])
    print("Последние слова текста:", phrase)
    print("Предсказанное следующее слово:", model.predict(phrase))
