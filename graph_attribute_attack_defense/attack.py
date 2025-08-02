import torch
import torch.nn.functional as F

def pgd_attack(model, data, nodes_to_attack, epsilon, num_iterations, learning_rate):
    """
    Implements the PGD attack on node features.
    """
    # Get the original features
    original_features = data.x.clone().detach()
    perturbed_features = data.x.clone().detach()
    
    for _ in range(num_iterations):
        # We need gradients w.r.t. the features
        perturbed_features.requires_grad = True
        
        # Forward pass
        output = model(data.__class__(x=perturbed_features, edge_index=data.edge_index))
        
        # Calculate loss only for the nodes we are attacking
        loss = F.nll_loss(output[nodes_to_attack], data.y[nodes_to_attack])
        
        # Backward pass to get gradients
        model.zero_grad()
        loss.backward()
        
        # Get the gradients
        grad = perturbed_features.grad.data[nodes_to_attack]
        
        # Update the features of the attacked nodes
        perturbation = learning_rate * grad.sign()
        
        # Add perturbation
        perturbed_features.data[nodes_to_attack] += perturbation
        
        # Project the perturbation back to the epsilon-ball
        diff = perturbed_features.data[nodes_to_attack] - original_features[nodes_to_attack]
        diff = torch.clamp(diff, -epsilon, epsilon)
        
        # Apply the projected perturbation
        perturbed_features.data[nodes_to_attack] = original_features[nodes_to_attack] + diff
        
        # Detach for the next iteration
        perturbed_features = perturbed_features.detach()

    return perturbed_features
