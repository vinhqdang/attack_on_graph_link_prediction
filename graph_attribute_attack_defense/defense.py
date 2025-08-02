import torch
import torch.nn.functional as F

def gfat_perturbation(model, data, delta, xi=1e-6):
    """
    Calculates the adversarial perturbation for GFAT.
    """
    # Generate a random unit vector
    d = torch.randn_like(data.x)
    d = F.normalize(d, p=2, dim=1)

    # Temporarily enable gradients for the input features
    data.x.requires_grad_()

    # Forward pass with a small perturbation xi * d
    output = model(data.__class__(x=data.x + xi * d, edge_index=data.edge_index))
    
    # Calculate the loss
    loss = F.nll_loss(output, data.y)

    # Backward pass to get gradients w.r.t. the small perturbation
    model.zero_grad()
    loss.backward()

    # The gradient is the direction of the adversarial perturbation
    grad = data.x.grad.data.detach()
    
    # Disable gradients for the input features again
    data.x.requires_grad_(False)

    # Calculate the final perturbation
    perturbation = delta * F.normalize(grad, p=2, dim=1)

    return perturbation
