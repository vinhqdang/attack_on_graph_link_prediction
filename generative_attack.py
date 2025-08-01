import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, VGAE
from torch_geometric.utils import dense_to_sparse, to_dense_adj, train_test_split_edges
from sklearn.metrics import roc_auc_score, average_precision_score

# Define the GNN-based attacker
class GNNAtk(nn.Module):
    def __init__(self, n_feat, n_hid):
        super(GNNAtk, self).__init__()
        self.gcn1 = GCNConv(n_feat, n_hid)
        self.gcn2 = GCNConv(n_hid, n_hid)

    def forward(self, x, adj):
        x = F.relu(self.gcn1(x, adj))
        x = self.gcn2(x, adj)
        # Generate a probability for each potential edge
        edge_probs = torch.sigmoid(torch.matmul(x, x.t()))
        return edge_probs.view(-1, 1)

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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    name_dataset = 'Cora'
    dataset = Planetoid(root='/tmp/' + name_dataset, name=name_dataset, transform=T.NormalizeFeatures())
    data = dataset[0]
    data = train_test_split_edges(data).to(device)

    n_nodes = data.num_nodes
    n_features = data.num_features
    out_channels = 16
    
    # The target VGAE model
    vgae_model = CustomVGAE(VariationalGCNEncoder(n_features, out_channels)).to(device)

    # The GNN attacker
    attacker = GNNAtk(n_features, 32).to(device)
    optimizer_attacker = torch.optim.Adam(attacker.parameters(), lr=0.01)

    adj_orig = to_dense_adj(data.train_pos_edge_index, max_num_nodes=n_nodes)[0]

    print("--- Starting Generative Attack Training ---")
    for epoch in range(100):
        attacker.train()
        optimizer_attacker.zero_grad()

        # Generate edge probabilities with the attacker
        edge_probs = attacker(data.x, data.train_pos_edge_index)
        
        # Sample a discrete adjacency matrix using Gumbel-Softmax
        edge_probs_cat = torch.cat([1-edge_probs, edge_probs], dim=-1)
        perturbed_adj = F.gumbel_softmax(edge_probs_cat, tau=1, hard=True)[:, 1].view(n_nodes, n_nodes)
        
        # Make the graph undirected
        perturbed_adj = perturbed_adj + perturbed_adj.T
        perturbed_adj[perturbed_adj > 1] = 1
        
        perturbed_edge_index, perturbed_edge_weight = dense_to_sparse(perturbed_adj)

        # Calculate the loss for the target model
        z = vgae_model.encode(data.x, perturbed_edge_index, edge_weight=perturbed_edge_weight)
        loss = -vgae_model.recon_loss(z, data.train_pos_edge_index) # Maximize the loss
        
        loss.backward()
        optimizer_attacker.step()

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Attack Loss: {loss.item():.4f}')

    print("--- Generative Attack Training Finished ---")

    # --- Evaluation Phase ---
    print("\n--- Evaluating Attack Performance ---")
    attacker.eval()
    edge_probs = attacker(data.x, data.train_pos_edge_index)
    edge_probs_cat = torch.cat([1-edge_probs, edge_probs], dim=-1)
    final_poisoned_adj = F.gumbel_softmax(edge_probs_cat, tau=1, hard=True)[:, 1].view(n_nodes, n_nodes).detach()
    final_poisoned_adj = final_poisoned_adj + final_poisoned_adj.T
    final_poisoned_adj[final_poisoned_adj > 1] = 1
    
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

    print("\n--- Generative Attack Results ---")
    print(f"VGAE on Generative Poisoned Graph AUC: {final_auc:.4f}")

if __name__ == '__main__':
    main()
