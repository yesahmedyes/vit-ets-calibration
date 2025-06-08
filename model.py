import torch
import torch.nn as nn
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
from typing import List


class ViTModel(nn.Module):
    def __init__(
        self, model_name: str = "google/vit-base-patch16-224", num_classes: int = 100
    ):
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes

        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTForImageClassification.from_pretrained(
            model_name, num_labels=num_classes, ignore_mismatched_sizes=True
        )

        self.freeze_backbone()

        self.model.eval()

    def freeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def unfreeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=pixel_values)

        return outputs.logits

    def predict_proba(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(images)
            probabilities = torch.softmax(logits, dim=-1)

        return probabilities

    def predict_logits(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(images)

        return logits

    def preprocess_images(self, images: List[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images, return_tensors="pt")

        return inputs["pixel_values"]

    def to(self, device):
        self.model = self.model.to(device)

        return self

    def train(self, mode: bool = True):
        self.model.train(mode)

        return self

    def eval(self):
        self.model.eval()

        return self
