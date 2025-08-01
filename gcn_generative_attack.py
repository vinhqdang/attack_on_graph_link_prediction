import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dense_to_sparse, train_test_split_edges
from sklearn.metrics import roc_auc_score, average_precision_score

# GCN Model (The Target)
class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 128)
        self.conv2 = GCNConv(128, out_channels)

    def encode(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight=edge_weight).relu()
        return self.conv2(x, edge_index, edge_weight=edge_weight)

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

# Generative GNN Attacker
class GNNAtk(nn.Module):
    def __init__(self, n_feat, n_hid):
        super(GNNAtk, self).__init__()
        self.gcn1 = GCNConv(n_feat, n_hid)
        self.gcn2 = GCNConv(n_hid, n_hid)

    def forward(self, x, adj):
        x = F.relu(self.gcn1(x, adj))
        x = self.gcn2(x, adj)
        edge_probs = torch.sigmoid(torch.matmul(x, x.t()))
        return edge_probs.view(-1, 1)

def get_link_labels(pos_edge_index, neg_edge_index):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

def test(model, data):
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.train_pos_edge_index)
        
    pos_edge_index = data.test_pos_edge_index
    neg_edge_index = data.test_neg_edge_index
    
    link_logits = model.decode(z, pos_edge_index, neg_edge_index)
    link_labels = get_link_labels(pos_edge_index, neg_edge_index).to(data.x.device)
    
    link_probs = link_logits.sigmoid()
    
    return roc_auc_score(link_labels.cpu(), link_probs.cpu()), average_precision_score(link_labels.cpu(), link_probs.cpu())


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    name_dataset = 'Cora'
    dataset = Planetoid(root='/tmp/' + name_dataset, name=name_dataset, transform=T.NormalizeFeatures())
    data = dataset[0]
    data = train_test_split_edges(data).to(device)

    n_nodes = data.num_nodes
    n_features = data.num_features
    
    # --- Pre-train a target GCN model ---
    print("--- Pre-training Target GCN Model ---")
    target_gcn = GCN(n_features, 64).to(device)
    optimizer_gcn = torch.optim.Adam(target_gcn.parameters(), lr=0.01)
    for epoch in range(100):
        target_gcn.train()
        optimizer_gcn.zero_grad()
        z = target_gcn.encode(data.x, data.train_pos_edge_index)
        neg_edge_index = torch.randint(0, data.num_nodes, (2, data.train_pos_edge_index.size(1)), dtype=torch.long, device=device)
        link_logits = target_gcn.decode(z, data.train_pos_edge_index, neg_edge_index)
        link_labels = get_link_labels(data.train_pos_edge_index, neg_edge_index).to(device)
        loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
        loss.backward()
        optimizer_gcn.step()
    target_gcn.eval()

    # --- Train the Generative Attacker ---
    print("\n--- Training Generative GNN Attacker ---")
    attacker = GNNAtk(n_features, 32).to(device)
    optimizer_attacker = torch.optim.Adam(attacker.parameters(), lr=0.01)

    for epoch in range(100):
        attacker.train()
        optimizer_attacker.zero_grad()

        edge_probs = attacker(data.x, data.train_pos_edge_index)
        edge_probs_cat = torch.cat([1-edge_probs, edge_probs], dim=-1)
        
        perturbed_adj_dist = F.gumbel_softmax(edge_probs_cat, tau=1, hard=True)[:, 1].view(n_nodes, n_nodes)
        perturbed_adj = perturbed_adj_dist + perturbed_adj_dist.T
        perturbed_adj[perturbed_adj > 1] = 1
        
        perturbed_edge_index, _ = dense_to_sparse(perturbed_adj)
        
        z = target_gcn.encode(data.x, perturbed_edge_index)
        neg_edge_index = torch.randint(0, data.num_nodes, (2, data.train_pos_edge_index.size(1)), dtype=torch.long, device=device)
        link_logits = target_gcn.decode(z, data.train_pos_edge_index, neg_edge_index)
        link_labels = get_link_labels(data.train_pos_edge_index, neg_edge_index).to(device)
        
        loss = -F.binary_cross_entropy_with_logits(link_logits, link_labels) # Maximize the loss
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
    final_poisoned_adj_dist = F.gumbel_softmax(edge_probs_cat, tau=1, hard=True)[:, 1].view(n_nodes, n_nodes).detach()
    final_poisoned_adj = final_poisoned_adj_dist + final_poisoned_adj_dist.T
    final_poisoned_adj[final_poisoned_adj > 1] = 1
    
    final_poisoned_edge_index, _ = dense_to_sparse(final_poisoned_adj)

    print("Training GCN on the POISONED graph...")
    poisoned_gcn = GCN(n_features, 64).to(device)
    optimizer_poisoned = torch.optim.Adam(poisoned_gcn.parameters(), lr=0.01)

    for epoch in range(200):
        poisoned_gcn.train()
        optimizer_poisoned.zero_grad()
        z = poisoned_gcn.encode(data.x, final_poisoned_edge_index)
        neg_edge_index = torch.randint(0, data.num_nodes, (2, final_poisoned_edge_index.size(1)), dtype=torch.long, device=device)
        link_logits = poisoned_gcn.decode(z, final_poisoned_edge_index, neg_edge_index)
        link_labels = get_link_labels(final_poisoned_edge_index, neg_edge_index).to(device)
        loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
        loss.backward()
        optimizer_poisoned.step()

    poisoned_gcn.eval()
    with torch.no_grad():
        z = poisoned_gcn.encode(data.x, data.train_pos_edge_index)
    
    test_auc, test_ap = test(poisoned_gcn, data)
    print(f'\n--- GCN on Poisoned Graph Final Performance ---')
    print(f'Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}')

if __name__ == '__main__':
    main()
