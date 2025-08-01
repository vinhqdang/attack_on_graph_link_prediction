import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
from sklearn.metrics import roc_auc_score, average_precision_score

class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, 128)
        self.conv2 = GCNConv(128, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return prob_adj

def get_link_labels(pos_edge_index, neg_edge_index):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    z = model.encode(data.x, data.train_pos_edge_index)
    
    # We perform a new round of negative sampling for every training epoch:
    neg_edge_index = torch.randint(0, data.num_nodes, (2, 5000), dtype=torch.long, device=data.x.device)
    
    link_logits = model.decode(z, data.train_pos_edge_index, neg_edge_index)
    link_labels = get_link_labels(data.train_pos_edge_index, neg_edge_index).to(data.x.device)
    
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
    loss.backward()
    optimizer.step()
    return loss

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

    model = GCN(data.num_features, 64).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

    print("--- Training GCN Baseline Model ---")
    for epoch in range(1, 201):
        loss = train(model, data, optimizer)
        if epoch % 20 == 0:
            val_auc, val_ap = test(model, data)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}, Val AP: {val_ap:.4f}')

    test_auc, test_ap = test(model, data)
    print(f'\n--- GCN Baseline Final Performance ---')
    print(f'Test AUC: {test_auc:.4f}, Test AP: {test_ap:.4f}')

if __name__ == '__main__':
    main()
