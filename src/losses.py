
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

#------------------------------------------------------------------------------
#
#   VanillaGanLoss
#
#------------------------------------------------------------------------------

class VanillaGanLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()
 
    def D(self, score, is_real):
        if (is_real):
            y = torch.ones_like(score)
        else:
            y = torch.zeros_like(score)
        return self.loss(score, y)

    def G(self, score):
        y = torch.ones_like(score)
        return self.loss(score, y)


#------------------------------------------------------------------------------
#
#   HingeGanLoss
#
#------------------------------------------------------------------------------

class HingeGanLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def D(self, score, is_real):
        if (is_real):
            loss = F.relu(1 - score).mean()
        else:
            loss = F.relu(1 + score).mean()
        return loss

    def G(self, score):
        return -score.mean()

#------------------------------------------------------------------------------
#
#   WassersteinGanLoss
#
#------------------------------------------------------------------------------

class WassersteinGanLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def D(self, score, is_real):
        if (is_real):
            loss = -score.mean()
        else:
            loss = score.mean()
        return loss

    def G(self, score):
        return -score.mean()

    def gradient_penalty(self, dis, x, x_fake):
        
        B = x.size(0)

        # Sample epsilon - Uniform(0, 1)
        epsilon = torch.rand(size=(B,1,1,1), device=x.device)

        # X_Hat - interpolated between real and fake X
        x_hat = epsilon*x + (1-epsilon)*x_fake.detach()
        x_hat.requires_grad = True

        # Get discriminator scores for X_Hat
        score_xhat = dis(x_hat)

        # Calculate gradients of X_Hat
        grad_xhat = grad(outputs=score_xhat.sum(), inputs=x_hat, create_graph=True)[0]
        
        # Flatten, and compute L-2 norm for each sample in batch
        grad_xhat_norm = grad_xhat.view(B,-1).norm(2, dim=1)

        # Penalize if gradient norm is not 1.0
        grad_penalty = (grad_xhat_norm - 1) ** 2

        # Return mean GP over batch
        return grad_penalty.mean()


def create_loss(args):

    LOSSES = {
        "gan": VanillaGanLoss,
        "hinge": HingeGanLoss,
        "wgan-gp": WassersteinGanLoss
    }

    return LOSSES[args.loss]()
