import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges, to_dense_adj
from sklearn.metrics import roc_auc_score, average_precision_score

class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True)
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

class VGAE(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGAE, self).__init__()
        self.encoder = VariationalGCNEncoder(in_channels, out_channels)

    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        return mu

    def encode(self, x, edge_index):
        self.mu, self.logstd = self.encoder(x, edge_index)
        z = self.reparametrize(self.mu, self.logstd)
        return z

    def decode(self, z):
        return torch.sigmoid(torch.matmul(z, z.t()))

    def recon_loss(self, z, adj_target):
        reconstructed_adj = self.decode(z)
        return F.binary_cross_entropy(reconstructed_adj, adj_target)

    def kl_loss(self, mu=None, logstd=None):
        mu = self.mu if mu is None else mu
        logstd = self.logstd if logstd is None else logstd.clamp(max=10)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1)
        )

    def test(self, z, pos_edge_index, neg_edge_index):
        reconstructed_adj = self.decode(z)
        pos_preds = reconstructed_adj[pos_edge_index[0], pos_edge_index[1]]
        neg_preds = reconstructed_adj[neg_edge_index[0], neg_edge_index[1]]
        
        preds = torch.cat([pos_preds, neg_preds], dim=0)
        labels = torch.cat([torch.ones_like(pos_preds), torch.zeros_like(neg_preds)], dim=0)
        
        preds, labels = preds.detach().cpu().numpy(), labels.detach().cpu().numpy()
        return roc_auc_score(labels, preds), average_precision_score(labels, preds)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0].to(device)
    data = train_test_split_edges(data)

    model = VGAE(dataset.num_features, 16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    adj = to_dense_adj(data.train_pos_edge_index, max_num_nodes=data.num_nodes)[0]

    def train():
        model.train()
        optimizer.zero_grad()
        z = model.encode(data.x, data.train_pos_edge_index)
        loss = model.recon_loss(z, adj) + (1 / data.num_nodes) * model.kl_loss()
        loss.backward()
        optimizer.step()
        return float(loss)

    def test(pos_edge_index, neg_edge_index):
        model.eval()
        with torch.no_grad():
            z = model.encode(data.x, data.train_pos_edge_index)
        return model.test(z, pos_edge_index, neg_edge_index)

    for epoch in range(1, 201):
        loss = train()
        auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
        print(f'Epoch: {epoch:03d}, AUC: {auc:.4f}, AP: {ap:.4f}, Loss: {loss:.4f}')

if __name__ == '__main__':
    main()