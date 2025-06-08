import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class TemperatureScaling(nn.Module):
    def __init__(self):
        super().__init__()

        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=1e-6)

    def calibrate(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> float:
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()

        def eval_loss():
            optimizer.zero_grad()

            with torch.no_grad():
                self.temperature.copy_(self.temperature.clamp(min=1e-6))

            loss = criterion(self.forward(logits), labels)
            loss.backward()

            return loss

        optimizer.step(eval_loss)

        with torch.no_grad():
            self.temperature.copy_(self.temperature.clamp(min=1e-6))


class EnsembleTemperatureScaling(nn.Module):
    def __init__(self):
        super().__init__()

        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self.weights = nn.Parameter(torch.ones(3) / 3)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # Component 1: Original TS calibration T(z; t)
        component1 = logits / self.temperature.clamp(min=1e-6)

        # Component 2: Uncalibrated prediction (t=1)
        component2 = logits.clone()

        # Component 3: Uniform prediction (t=∞, outputs 1/L for each class)
        _, num_classes = logits.shape
        component3 = torch.ones_like(logits) / num_classes

        calibrated_probs = (
            self.weights[0] * component1
            + self.weights[1] * component2
            + self.weights[2] * component3
        )

        return calibrated_probs

    def calibrate(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> np.ndarray:
        optimizer = optim.LBFGS(
            [self.temperature, self.weights], lr=lr, max_iter=max_iter
        )
        criterion = nn.CrossEntropyLoss()

        def eval_loss():
            optimizer.zero_grad()

            with torch.no_grad():
                self.temperature.copy_(self.temperature.clamp(min=1e-6))
                self.weights.copy_(torch.softmax(self.weights, dim=0))

            loss = criterion(self.forward(logits), labels)
            loss.backward()

            return loss

        optimizer.step(eval_loss)

        with torch.no_grad():
            self.temperature.copy_(self.temperature.clamp(min=1e-6))
            self.weights.copy_(torch.softmax(self.weights, dim=0))
