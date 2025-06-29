import random
import torch
import torch.nn.functional as F
from torch import nn

try:
    from torch_geometric.datasets import AIFB
    from torch_geometric.nn import RGCNConv
except ImportError as e:
    raise ImportError("This script requires PyTorch Geometric. Please install it before running.") from e


def load_dataset():
    dataset = AIFB(root='data/AIFB')
    data = dataset[0]
    num_relations = int(data.edge_type.max().item()) + 1
    return dataset, data, num_relations


class RelGCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_relations):
        super().__init__()
        self.conv1 = RGCNConv(in_feats, hidden_feats, num_relations)
        self.conv2 = RGCNConv(hidden_feats, out_feats, num_relations)

    def forward(self, x, edge_index, edge_type):
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = self.conv2(x, edge_index, edge_type)
        return x


def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, data.edge_type)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()


def test(model, data):
    model.eval()
    out = model(data.x, data.edge_index, data.edge_type)
    loss = F.cross_entropy(out[data.test_mask], data.y[data.test_mask]).item()
    pred = out[data.test_mask].argmax(dim=1)
    acc = (pred == data.y[data.test_mask]).sum().item() / int(data.test_mask.sum())
    return acc, loss


def flip_edge(edge_index, edge_type, num_relations, u, v):
    mask = ~(((edge_index[0] == u) & (edge_index[1] == v)) |
             ((edge_index[0] == v) & (edge_index[1] == u)))
    if mask.all():
        rel = random.randrange(num_relations)
        new_edges = torch.cat([
            edge_index, torch.tensor([[u, v], [v, u]], dtype=torch.long, device=edge_index.device)
        ], dim=1)
        new_types = torch.cat([
            edge_type, torch.tensor([rel, rel], dtype=torch.long, device=edge_index.device)
        ])
    else:
        new_edges = edge_index[:, mask]
        new_types = edge_type[mask]
    return new_edges, new_types


def flip_feature(features, node, idx):
    features[node, idx] = 1 - features[node, idx]
    return features


def run_attack(model, data, num_relations, num_target_nodes=100, budget=10):
    model.eval()
    logits = model(data.x, data.edge_index, data.edge_type)
    conf = F.softmax(logits[data.test_mask], dim=1)
    conf_max, pred = conf.max(dim=1)
    true = data.y[data.test_mask]
    correct_mask = (pred == true)
    target_candidates = data.test_mask.nonzero(as_tuple=False).view(-1)[correct_mask]
    target_conf = conf_max[correct_mask]
    _, sorted_idx = torch.sort(target_conf, descending=True)
    target_nodes = target_candidates[sorted_idx][:num_target_nodes].tolist()

    successful = 0
    modified_links = 0
    for node in target_nodes:
        for _ in range(budget):
            best_loss = -float('inf')
            best_action = None
            neighbors = random.sample(range(data.num_nodes), min(20, data.num_nodes))
            features = random.sample(range(data.num_node_features), min(20, data.num_node_features))
            for n in neighbors:
                tmp_edge, tmp_type = flip_edge(data.edge_index, data.edge_type, num_relations, node, n)
                out = model(data.x, tmp_edge, tmp_type)
                loss = F.cross_entropy(out[[node]], data.y[[node]]).item()
                if loss > best_loss:
                    best_loss = loss
                    best_action = ('edge', n)
            for f in features:
                tmp_x = data.x.clone()
                tmp_x = flip_feature(tmp_x, node, f)
                out = model(tmp_x, data.edge_index, data.edge_type)
                loss = F.cross_entropy(out[[node]], data.y[[node]]).item()
                if loss > best_loss:
                    best_loss = loss
                    best_action = ('feature', f)
            if best_action is None:
                continue
            if best_action[0] == 'edge':
                data.edge_index, data.edge_type = flip_edge(
                    data.edge_index, data.edge_type, num_relations, node, best_action[1])
                modified_links += 1
            else:
                data.x = flip_feature(data.x, node, best_action[1])

        out = model(data.x, data.edge_index, data.edge_type)
        pred = out[[node]].argmax(dim=1)
        if pred.item() != data.y[node].item():
            successful += 1

    acc_after, loss_after = test(model, data)
    asr = successful / len(target_nodes) * 100
    aml = modified_links / len(target_nodes)
    return acc_after, loss_after, asr, aml


def main():
    random.seed(42)
    torch.manual_seed(42)

    dataset, data, num_relations = load_dataset()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    model = RelGCN(dataset.num_features, 16, dataset.num_classes, num_relations).to(device)
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
