import torch
import torch.nn.functional as F

def gfat_perturbation(model, data, delta, xi=1e-6):
    """
    Calculates the adversarial perturbation for GFAT using Virtual Adversarial Training.
    This prevents data leakage from the test set.
    """
    # Get the model's current predictions and detach them to use as fixed targets.
    with torch.no_grad():
        clean_output = model(data)
    target_dist = clean_output.detach()

    # Generate a random unit vector
    d = torch.randn_like(data.x)
    d = F.normalize(d, p=2, dim=1)

    # Temporarily enable gradients for the input features
    data.x.requires_grad_()

    # Forward pass with a small perturbation xi * d
    perturbed_output = model(data.__class__(x=data.x + xi * d, edge_index=data.edge_index))
    
    # Calculate the KL-divergence between the perturbed output and the target distribution.
    # We use log_softmax for numerical stability.
    loss = F.kl_div(F.log_softmax(perturbed_output, dim=1), F.softmax(target_dist, dim=1), reduction='batchmean')

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
