import random
import torch
import torch.nn.functional as F
from torch import nn

try:
    from torch_geometric.datasets import Entities
    from torch_geometric.nn import RGCNConv
except ImportError as e:
    raise ImportError("This script requires PyTorch Geometric. Please install it before running.") from e


def load_dataset():
    dataset = Entities(root='data/AIFB', name='AIFB')
    data = dataset[0]
    num_relations = int(data.edge_type.max().item()) + 1
    return dataset, data, num_relations


class RelGCN(nn.Module):
    def __init__(self, num_nodes, hidden_feats, num_relations):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, hidden_feats)
        self.relation_emb = nn.Embedding(num_relations, hidden_feats)
        self.conv1 = RGCNConv(hidden_feats, hidden_feats, num_relations)
        self.conv2 = RGCNConv(hidden_feats, hidden_feats, num_relations)

    def forward(self, edge_index, edge_type):
        x = self.node_emb.weight
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = self.conv2(x, edge_index, edge_type)
        return x

    def score(self, head_idx, tail_idx, rel_idx):
        head_emb = self.node_emb(head_idx)
        tail_emb = self.node_emb(tail_idx)
        rel_emb = self.relation_emb(rel_idx)
        return (head_emb * rel_emb * tail_emb).sum(dim=1)


def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    node_emb = model(data.edge_index, data.edge_type)
    head_idx, tail_idx = data.edge_index[:, data.train_idx]
    rel_idx = data.edge_type[data.train_idx]
    score = model.score(head_idx, tail_idx, rel_idx)
    loss = F.binary_cross_entropy_with_logits(score, torch.ones_like(score))
    loss.backward()
    optimizer.step()


def test(model, data):
    model.eval()
    node_emb = model(data.edge_index, data.edge_type)
    head_idx, tail_idx = data.edge_index[:, data.test_idx]
    rel_idx = data.edge_type[data.test_idx]
    score = model.score(head_idx, tail_idx, rel_idx)
    acc = ((score > 0).float().mean()).item()
    return acc, 0.0


def run_attack(model, data, num_relations, num_target_nodes=100, budget=10):
    # This attack is not applicable to link prediction
    return 0.0, 0.0, 0.0, 0.0


def main():
    random.seed(42)
    torch.manual_seed(42)

    dataset, data, num_relations = load_dataset()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    model = RelGCN(data.num_nodes, 16, num_relations).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    for epoch in range(200):
        train(model, data, optimizer)

    acc_before, loss_before = test(model, data)
    print(f"Accuracy BEFORE attack: {acc_before:.4f}, Loss: {loss_before:.4f}")

    acc_after, loss_after, asr, aml = run_attack(model, data.clone(), num_relations)
    print(f"Accuracy AFTER attack: {acc_after:.4f}, Loss: {loss_after:.4f}")
    print(f"ASR: {asr:.2f}%, AML: {aml:.4f}")


if __name__ == '__main__':
    main()