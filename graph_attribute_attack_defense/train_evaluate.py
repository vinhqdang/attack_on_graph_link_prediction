import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import degree

from gcn_model import GCN
from attack import pgd_attack
from defense import gfat_perturbation

def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

def train_gfat(model, data, optimizer, delta):
    model.train()
    optimizer.zero_grad()
    
    # Get adversarial perturbation
    perturbation = gfat_perturbation(model, data, delta)
    
    # Apply perturbation for training
    perturbed_data = data.__class__(x=data.x + perturbation, edge_index=data.edge_index, y=data.y)
    perturbed_data.train_mask = data.train_mask

    output = model(perturbed_data)
    loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

def test(model, data):
    model.eval()
    output = model(data)
    pred = output.argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    return acc

def main():
    # --- Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0].to(device)
    
    num_node_features = dataset.num_node_features
    num_classes = dataset.num_classes

    # --- Scenario 1: Standard GCN on Clean Data ---
    print("--- Scenario 1: Training standard GCN on clean data ---")
    standard_model_clean = GCN(num_node_features, num_classes).to(device)
    optimizer_standard_clean = torch.optim.Adam(standard_model_clean.parameters(), lr=0.01, weight_decay=5e-4)
    
    for epoch in range(200):
        train(standard_model_clean, data, optimizer_standard_clean)
    
    acc_standard_clean = test(standard_model_clean, data)
    print(f"Accuracy of standard GCN on clean data: {acc_standard_clean:.4f}\n")

    # --- Scenario 2: Baseline Robust GCN (GFAT on clean data) ---
    print("--- Scenario 2: Training robust GCN with GFAT on clean data ---")
    gfat_model_clean = GCN(num_node_features, num_classes).to(device)
    optimizer_gfat_clean = torch.optim.Adam(gfat_model_clean.parameters(), lr=0.01, weight_decay=5e-4)
    
    for epoch in range(200):
        train_gfat(gfat_model_clean, data, optimizer_gfat_clean, delta=0.1)
    
    acc_gfat_clean = test(gfat_model_clean, data)
    print(f"Accuracy of robust GCN (GFAT) on clean data: {acc_gfat_clean:.4f}\n")

    # --- Identify nodes to attack (top 50 degree nodes) ---
    node_degrees = degree(data.edge_index[0]).to(device)
    top_k_nodes = torch.topk(node_degrees, k=50).indices

    # --- Scenario 3: Generate Attacked Data ---
    print("--- Scenario 3: Generating attacked data using PGD ---")
    # First, train a standard GCN to be the surrogate model for the attacker
    surrogate_model = GCN(num_node_features, num_classes).to(device)
    optimizer_surrogate = torch.optim.Adam(surrogate_model.parameters(), lr=0.01, weight_decay=5e-4)
    for epoch in range(100):
        train(surrogate_model, data, optimizer_surrogate)
    
    # Generate perturbed features
    perturbed_features = pgd_attack(surrogate_model, data, top_k_nodes, epsilon=0.1, num_iterations=40, learning_rate=0.01)
    attacked_data = data.__class__(x=perturbed_features, edge_index=data.edge_index, y=data.y)
    attacked_data.train_mask = data.train_mask
    attacked_data.test_mask = data.test_mask
    print("Data attack complete.\n")

    # --- Scenario 4: Standard GCN on Attacked Data ---
    print("--- Scenario 4: Training standard GCN on attacked data ---")
    standard_model_attacked = GCN(num_node_features, num_classes).to(device)
    optimizer_standard_attacked = torch.optim.Adam(standard_model_attacked.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(200):
        train(standard_model_attacked, attacked_data, optimizer_standard_attacked)
        
    acc_standard_attacked = test(standard_model_attacked, attacked_data)
    print(f"Accuracy of standard GCN on attacked data: {acc_standard_attacked:.4f}\n")

    # --- Scenario 5: Robust GCN (GFAT) on Attacked Data ---
    print("--- Scenario 5: Training robust GCN (GFAT) on attacked data ---")
    gfat_model_attacked = GCN(num_node_features, num_classes).to(device)
    optimizer_gfat_attacked = torch.optim.Adam(gfat_model_attacked.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(200):
        train_gfat(gfat_model_attacked, attacked_data, optimizer_gfat_attacked, delta=0.1)

    acc_gfat_attacked = test(gfat_model_attacked, attacked_data)
    print(f"Accuracy of robust GCN (GFAT) on attacked data: {acc_gfat_attacked:.4f}\n")

    # --- Final Comparison ---
    print("--- Final Results ---")
    print(f"Standard GCN on Clean Data:       {acc_standard_clean:.4f}")
    print(f"Robust GCN (GFAT) on Clean Data:    {acc_gfat_clean:.4f}")
    print(f"Standard GCN on Attacked Data:      {acc_standard_attacked:.4f}")
    print(f"Robust GCN (GFAT) on Attacked Data: {acc_gfat_attacked:.4f}")

if __name__ == '__main__':
    main()