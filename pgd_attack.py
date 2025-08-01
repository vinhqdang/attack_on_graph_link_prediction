import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, VGAE
from torch_geometric.utils import dense_to_sparse, to_dense_adj, train_test_split_edges
from sklearn.metrics import roc_auc_score, average_precision_score

# Define the Encoder, which can handle weighted edges
class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        return self.conv_mu(x, edge_index, edge_weight=edge_weight), self.conv_logstd(x, edge_index, edge_weight=edge_weight)

# Re-define the VGAE model to include a test function
class CustomVGAE(VGAE):
    def __init__(self, encoder):
        super(CustomVGAE, self).__init__(encoder)

    def test(self, z, pos_edge_index, neg_edge_index):
        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        return roc_auc_score(y, pred), average_precision_score(y, pred)

def pgd_attack(model, data, epochs=50, budget=15, lr=0.01):
    """
    Performs a PGD attack on the graph structure.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_nodes = data.num_nodes
    adj_orig = to_dense_adj(data.train_pos_edge_index, max_num_nodes=n_nodes)[0]
    
    # Create a learnable perturbation
    adj_changes = torch.zeros(n_nodes, n_nodes, requires_grad=True, device=device)

    for epoch in range(epochs):
        # Calculate the gradient of the loss w.r.t. the perturbation
        perturbed_adj = adj_orig + adj_changes
        perturbed_edge_index, perturbed_edge_weight = dense_to_sparse(perturbed_adj)
        
        z = model.encode(data.x, perturbed_edge_index, edge_weight=perturbed_edge_weight)
        loss = -model.recon_loss(z, data.train_pos_edge_index) # Maximize the loss
        
        grad = torch.autograd.grad(loss, adj_changes)[0]

        # Update the perturbation
        adj_changes = adj_changes + lr * grad
        
        # Project the perturbation back into the budget
        adj_changes = torch.clamp(adj_changes, -1, 1)
        adj_changes = adj_changes * budget / torch.norm(adj_changes)

    # Get the final poisoned graph
    final_poisoned_adj = adj_orig + adj_changes.detach()
    final_poisoned_adj = torch.clamp(final_poisoned_adj, 0, 1) # Ensure valid adjacency matrix
    
    return dense_to_sparse(final_poisoned_adj)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    name_dataset = 'Cora'
    dataset = Planetoid(root='/tmp/' + name_dataset, name=name_dataset, transform=T.NormalizeFeatures())
    data = dataset[0]
    data = train_test_split_edges(data).to(device)

    n_nodes = data.num_nodes
    n_features = data.num_features
    out_channels = 16

    # Train a clean VGAE model
    print("--- Training Clean VGAE Model ---")
    clean_vgae = CustomVGAE(VariationalGCNEncoder(n_features, out_channels)).to(device)
    optimizer_clean = torch.optim.Adam(clean_vgae.parameters(), lr=0.01)
    for epoch in range(200):
        clean_vgae.train()
        optimizer_clean.zero_grad()
        z = clean_vgae.encode(data.x, data.train_pos_edge_index)
        loss = clean_vgae.recon_loss(z, data.train_pos_edge_index) + (1 / n_nodes) * clean_vgae.kl_loss()
        loss.backward()
        optimizer_clean.step()

    # --- PGD Attack ---
    print("\n--- Performing PGD Attack ---")
    poisoned_edge_index, poisoned_edge_weight = pgd_attack(clean_vgae, data)

    # Train a new VGAE model from scratch on the POISONED graph
    print("\n--- Training VGAE on the POISONED graph (PGD) ---")
    poisoned_vgae = CustomVGAE(VariationalGCNEncoder(n_features, out_channels)).to(device)
    optimizer_poisoned = torch.optim.Adam(poisoned_vgae.parameters(), lr=0.01)

    for epoch in range(200):
        poisoned_vgae.train()
        optimizer_poisoned.zero_grad()
        
        z = poisoned_vgae.encode(data.x, poisoned_edge_index, edge_weight=poisoned_edge_weight)
        loss = poisoned_vgae.recon_loss(z, poisoned_edge_index) + (1 / n_nodes) * poisoned_vgae.kl_loss()
        loss.backward()
        optimizer_poisoned.step()

    poisoned_vgae.eval()
    with torch.no_grad():
        z = poisoned_vgae.encode(data.x, poisoned_edge_index, edge_weight=poisoned_edge_weight)
        final_auc, final_ap = poisoned_vgae.test(z, data.test_pos_edge_index, data.test_neg_edge_index)

    print("\n--- PGD Attack Results ---")
    print(f"VGAE on PGD Poisoned Graph AUC: {final_auc:.4f}")

if __name__ == '__main__':
    main()
