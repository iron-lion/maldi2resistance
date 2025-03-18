import torch


class MaskedBCE:
    def __init__(self, class_weights_positive=None, class_weights_negative=None, label_weights=None):
        self.class_weights_positive = class_weights_positive
        self.class_weights_negative = class_weights_negative
        self.label_weights = label_weights

    def __call__(self, prediction, truth):
        positive_weight = torch.clone(truth)
        negative_weight = torch.clone(truth)

        negative_weight[negative_weight == 1] = -1
        negative_weight[negative_weight == 0] = 1
        negative_weight[negative_weight == -1] = 0

        if self.class_weights_positive is not None:
            positive_weight = positive_weight * self.class_weights_positive[None, :]

        if self.class_weights_negative is not None:
            negative_weight = negative_weight * self.class_weights_negative[None, :]

        weight = torch.add(positive_weight, negative_weight)
        weight = torch.nan_to_num(weight, 0)

        if self.label_weights is not None:
            weight = weight * self.label_weights[None, :]

        try:
            weight = weight[:, 0, :]
        except IndexError:
            pass

        truth = torch.nan_to_num(truth, 0)

        loss = torch.nn.functional.binary_cross_entropy(
            prediction, truth, weight=weight
        )

        return loss
