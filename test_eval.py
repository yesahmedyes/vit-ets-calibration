import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from tqdm import tqdm


def get_logits_and_labels(
    model, dataloader, device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, total=len(dataloader)):
            images, labels = images.to(device), labels.to(device)

            if hasattr(model, "predict_logits"):
                logits = model.predict_logits(images)
            else:
                logits = model(images)

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def calculate_accuracy(probabilities: torch.Tensor, labels: torch.Tensor) -> float:
    predicted_classes = torch.argmax(probabilities, dim=-1)
    accuracy = (predicted_classes == labels).float().mean().item()

    return accuracy


def calculate_confidence(probabilities: torch.Tensor) -> float:
    confidences = torch.max(probabilities, dim=-1)[0]
    avg_confidence = confidences.mean().item()

    return avg_confidence


def expected_calibration_error(
    probabilities: torch.Tensor, labels: torch.Tensor, n_bins: int = 15
) -> float:
    probabilities = probabilities.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    predicted_classes = np.argmax(probabilities, axis=-1)
    confidences = np.max(probabilities, axis=-1)
    accuracies = predicted_classes == labels

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


def get_all_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    probs = torch.softmax(logits, dim=-1)

    accuracy = calculate_accuracy(probs, labels)
    confidence = calculate_confidence(probs)
    ece = expected_calibration_error(probs, labels)

    return {"accuracy": accuracy, "confidence": confidence, "ece": ece}


def reliability_diagram(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
    title: str = "Reliability Diagram",
):
    probabilities = torch.stack([torch.softmax(each, dim=-1) for each in logits])

    labels = labels.detach().cpu().numpy()
    probabilities = probabilities.detach().cpu().numpy()

    predicted_classes = np.argmax(probabilities, axis=-1)
    confidences = np.max(probabilities, axis=-1)
    accuracies = predicted_classes == labels

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    bin_centers = []
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            bin_centers.append((bin_lower + bin_upper) / 2)
            bin_accuracies.append(accuracies[in_bin].mean())
            bin_confidences.append(confidences[in_bin].mean())
            bin_counts.append(in_bin.sum())
        else:
            bin_centers.append((bin_lower + bin_upper) / 2)
            bin_accuracies.append(0)
            bin_confidences.append((bin_lower + bin_upper) / 2)
            bin_counts.append(0)

    plt.figure(figsize=(8, 6))

    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    plt.bar(
        bin_centers,
        bin_accuracies,
        width=1 / n_bins,
        alpha=0.7,
        edgecolor="black",
        label="Outputs",
    )

    gaps = np.array(bin_confidences) - np.array(bin_accuracies)
    plt.bar(
        bin_centers,
        gaps,
        bottom=bin_accuracies,
        width=1 / n_bins,
        alpha=0.5,
        color="red",
        edgecolor="black",
        label="Gap",
    )

    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
