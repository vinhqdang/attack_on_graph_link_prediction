import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, VGAE
from torch_geometric.utils import dense_to_sparse, to_dense_adj, train_test_split_edges
from sklearn.metrics import roc_auc_score, average_precision_score

# Enable anomaly detection to get a more detailed traceback
torch.autograd.set_detect_anomaly(True)

# Define the Encoder, which can handle weighted edges for differentiable attacks
class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        return self.conv_mu(x, edge_index, edge_weight=edge_weight), self.conv_logstd(x, edge_index, edge_weight=edge_weight)

# Define the MetaAttack model that learns a perturbation matrix
class MetaAttack(torch.nn.Module):
    def __init__(self, nnodes):
        super(MetaAttack, self).__init__()
        self.adj_changes = torch.nn.Parameter(torch.zeros(nnodes, nnodes))

    def forward(self, adj):
        adj_changes_symmetric = (self.adj_changes + self.adj_changes.T) / 2
        perturbed_adj = torch.sigmoid(adj + adj_changes_symmetric)
        return perturbed_adj

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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    name_dataset = 'Cora'
    dataset = Planetoid(root='/tmp/' + name_dataset, name=name_dataset, transform=T.NormalizeFeatures())
    data = dataset[0]
    data = train_test_split_edges(data).to(device)

    n_nodes = data.num_nodes
    n_features = data.num_features
    out_channels = 16

    # The VGAE model is the target of the attack
    vgae_model = CustomVGAE(VariationalGCNEncoder(n_features, out_channels)).to(device)
    
    # The attack model will learn to perturb the graph
    attack_model = MetaAttack(n_nodes).to(device)
    optimizer_attack = torch.optim.Adam(attack_model.parameters(), lr=0.01)

    # The original adjacency matrix
    adj_orig = to_dense_adj(data.train_pos_edge_index, max_num_nodes=n_nodes)[0]

    print("--- Starting Meta-Attack Training ---")
    for epoch in range(100):
        attack_model.train()
        vgae_model.train()
        optimizer_attack.zero_grad()
        
        perturbed_adj = attack_model(adj_orig)
        perturbed_edge_index, perturbed_edge_weight = dense_to_sparse(perturbed_adj)

        z = vgae_model.encode(data.x, perturbed_edge_index, edge_weight=perturbed_edge_weight)
        loss = vgae_model.recon_loss(z, data.train_pos_edge_index) + (1 / n_nodes) * vgae_model.kl_loss()
        
        attack_loss = -loss
        attack_loss.backward()
        optimizer_attack.step()

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Attack Loss: {attack_loss.item():.4f}')
    
    print("--- Meta-Attack Training Finished ---")
    
    # --- Evaluation Phase ---
    print("\n--- Evaluating Attack Performance ---")
    attack_model.eval()
    final_poisoned_adj = attack_model(adj_orig).detach()
    final_poisoned_edge_index, final_poisoned_edge_weight = dense_to_sparse(final_poisoned_adj)

    print("Training VGAE on the POISONED graph...")
    poisoned_vgae = CustomVGAE(VariationalGCNEncoder(n_features, out_channels)).to(device)
    optimizer_poisoned = torch.optim.Adam(poisoned_vgae.parameters(), lr=0.01)

    for epoch in range(200):
        poisoned_vgae.train()
        optimizer_poisoned.zero_grad()
        
        z = poisoned_vgae.encode(data.x, final_poisoned_edge_index, edge_weight=final_poisoned_edge_weight)
        loss = poisoned_vgae.recon_loss(z, final_poisoned_edge_index) + (1 / n_nodes) * poisoned_vgae.kl_loss()
        loss.backward()
        optimizer_poisoned.step()

    poisoned_vgae.eval()
    with torch.no_grad():
        z = poisoned_vgae.encode(data.x, final_poisoned_edge_index, edge_weight=final_poisoned_edge_weight)
        final_auc, final_ap = poisoned_vgae.test(z, data.test_pos_edge_index, data.test_neg_edge_index)

    print("\n--- Comparison Results ---")
    print(f"Baseline VGAE (clean graph) AUC: ~0.54 (from previous run)")
    print(f"VGAE on Poisoned Graph AUC: {final_auc:.4f}")
    print(f"The meta-attack changed the final AUC by: {final_auc - 0.54:.4f}")

if __name__ == '__main__':
    main()