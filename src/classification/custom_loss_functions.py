#!/usr/bin/env python
# coding: utf-8

""" Custom loss function definitions
"""
import torch
import torch.nn as nn


class WeightedOrderAndBinaryCrossEntropyLoss(nn.Module):
    """A custom loss function to compute a weighted cross entropy loss between
    order- and binary-level classification
    """

    def __init__(self, weight_on_order=0.5):
        """Class initiation"""

        super(WeightedOrderAndBinaryCrossEntropyLoss, self).__init__()
        self.weight_on_order = weight_on_order
        self.weight_on_binary = 1 - weight_on_order

    def _modify_multiclass_output_to_binary(
        self, prediction: torch.Tensor
    ) -> torch.Tensor:
        """Modify a multi-class output to a binary output"""

        # Apply softmax to get probabilities from logits
        softmax_probs = nn.functional.softmax(prediction, dim=1)

        # Combine probabilities for non-moth classes (index 1 and afterwards)
        combined_nonmoth_probs = softmax_probs[:, 1:].sum(dim=1)
        binary_probs = torch.cat(
            (
                torch.unsqueeze(softmax_probs[:, 0], 1),
                torch.unsqueeze(combined_nonmoth_probs, 1),
            ),
            dim=1,
        )

        # Recover logits from probabilities for cross-entropy loss
        # NOTE: The log of softmax does not recover logits exactly. The logits will
        # again be converted to softmax in cross-entropy loss, so the lack of
        # normalization constant doesn't make a difference here.
        binary_logits = torch.log(binary_probs)

        return binary_logits

    def forward(self, prediction: torch.Tensor, target: torch.Tensor):
        """Override the forward function to compute the weighted cross entropy loss.
        ASSUMPTION: The 0th label is Lepidoptera!
        """

        # Compute the order-level cross entropy loss
        order_loss = nn.CrossEntropyLoss()(prediction, target)

        # Modify targets to be 0 (Lepidoptera/moth) or 1 (non-moth)
        binary_target = torch.where(target != 0, torch.tensor(1), target)

        # Modify the multi-class predictions to be binary
        binary_prediction = self._modify_multiclass_output_to_binary(prediction)

        # Compute the binary-level cross entropy loss
        binary_loss = nn.CrossEntropyLoss()(binary_prediction, binary_target)

        # Compute the weighted average of the two losses
        weighted_loss = (
            self.weight_on_order * order_loss + self.weight_on_binary * binary_loss
        )

        return weighted_loss


if __name__ == "__main__":
    # Create sample input tensors and target tensors
    inputs = torch.randn(3, 3)  # Batch size of 3, 5 classes
    targets = torch.tensor([1, 0, 1])  # Target labels for first input

    # Initialize the custom loss function with weights
    criterion = WeightedOrderAndBinaryCrossEntropyLoss()

    # Compute the loss
    loss = criterion(inputs, targets)
    print(loss)
