import functools
import json
import os
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


def placeholder(func: Callable):
    @functools.wraps(func)
    def wrapped_function(*args, **kwargs):
        print(f"Executing placeholder {func.__name__} ... "
              f"(this may not have the actual behavior as per described in the paper)")
        return func(*args, **kwargs)
    return wrapped_function


class MLP(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 128, num_layers: int = 3,
                 noise_size: int = 64):
        super().__init__()
        in_sizes = [input_size + noise_size] + [hidden_size] * (num_layers - 1)
        out_sizes = [hidden_size] * (num_layers - 1) + [output_size]
        layers = [nn.Linear(in_sizes[0], out_sizes[0])]
        for i, o in zip(in_sizes[1:], out_sizes[1:]):
            layers.append(nn.Sequential(
                nn.ReLU(), nn.LayerNorm(i), nn.Dropout(0.1), nn.Linear(i, o)
            ))
        self._noise_size = noise_size
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn(x.shape[0], self._noise_size, device=x.device)
        return self.layers(torch.cat([x, noise], dim=-1))


def train_mlp(
        in_data: np.ndarray, out_data: np.ndarray, model_dir: str, epochs: int = 100, batch_size: int = 500, **kwargs
):
    os.makedirs(model_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLP(input_size=in_data.shape[1], output_size=out_data.shape[1], **kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.MSELoss()
    in_data = torch.from_numpy(in_data).float()
    out_data = torch.from_numpy(out_data).float()
    dataset = TensorDataset(in_data, out_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for i, o in dataloader:
            i = i.to(device)
            o = o.to(device)
            optimizer.zero_grad()
            output = model(i)
            loss = criterion(output, o)
            loss.backward()
            optimizer.step()

    with open(os.path.join(model_dir, "args.json"), "w") as f:
        json.dump(kwargs | {
            "input_size": in_data.shape[1], "output_size": out_data.shape[1], "batch_size": batch_size
        }, f)
    torch.save(model.state_dict(), os.path.join(model_dir, 'model.pt'))


def predict_mlp(
        in_data: np.ndarray, model_dir: str
) -> np.ndarray:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(model_dir, "args.json"), "r") as f:
        kwargs = json.load(f)
    batch_size = kwargs.pop('batch_size')
    model = MLP(**kwargs).to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt")))
    model.eval()
    in_data = torch.from_numpy(in_data).float()
    dataset = TensorDataset(in_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    out = []
    with torch.no_grad():
        for i, in dataloader:
            o = model(i.to(device))
            out.append(o)

    return torch.cat(out).cpu().numpy()
