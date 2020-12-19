import torch

from domain.base import Domain


class Criterion(Domain):

    def __init__(self):
        super(Criterion, self).__init__()

    @classmethod
    def mse_loss(cls, x, x_hat):
        return torch.mean(torch.pow((x - x_hat), 2))

    @classmethod
    def reconstruction_error(cls, x, x_hat):
        return ((x_hat - x) ** 2).mean(axis=1)

    @classmethod
    def sae_loss(cls, x, x_hat, latent_data, lambda_):
        return torch.mean(torch.pow((x - x_hat), 2)) + lambda_ * torch.mean(torch.pow(latent_data, 2))

    @classmethod
    def triple_margin_loss(cls, anchor, positive, negative, margin):
        triple_margin_loss = torch.nn.TripletMarginLoss(margin=margin)

        return triple_margin_loss(anchor=anchor, positive=positive, negative=negative)

    @classmethod
    def triple_margin_loss_custom(cls, positive_anchor, negative_anchor, positive, negative, margin):
        positive_distance = torch.mean(torch.pow((positive_anchor - positive), 2), dim=0)
        negative_distance = torch.mean(torch.pow((negative_anchor - negative), 2), dim=0)

        triple_margin_distance = margin + positive_distance - negative_distance

        return torch.mean(torch.max(torch.stack([triple_margin_distance,
                                                 torch.zeros(len(triple_margin_distance))], dim=1), dim=1).values)
