
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import degree

class GCN(torch.nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden_feats)
        self.conv2 = GCNConv(hidden_feats, out_feats)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

def train(model, data, epochs=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

def evaluate(model, data, mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct = (pred[mask] == data.y[mask]).sum().item()
        acc = correct / mask.sum().item()
        loss = F.cross_entropy(out[mask], data.y[mask]).item()
    return acc, loss

def select_target_nodes(data, strategy='high_degree', k=10):
    if strategy == 'high_degree':
        return degree(data.edge_index[0]).topk(k).indices
    elif strategy == 'low_confidence':
        # This requires a trained model to calculate confidence
        raise NotImplementedError("Low confidence strategy requires a trained model.")
    else:
        raise ValueError(f"Unknown target selection strategy: {strategy}")

def pgd_feature_attack(model, data, target_nodes, eps=0.1, alpha=0.01, iters=40):
    model.eval()
    x_adv = data.x.clone()
    x_adv.requires_grad = True

    for _ in range(iters):
        out = model(x_adv, data.edge_index)
        loss = F.cross_entropy(out[target_nodes], data.y[target_nodes])
        loss.backward()

        with torch.no_grad():
            grad = x_adv.grad.sign()
            x_adv.data[target_nodes] += alpha * grad[target_nodes]
            perturbation = torch.clamp(x_adv.data - data.x, -eps, eps)
            x_adv.data = data.x + perturbation
        x_adv.grad.zero_()

    return x_adv.detach()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Planetoid(root="/tmp/Cora", name="Cora")
    data = dataset[0].to(device)

    model = GCN(dataset.num_features, 16, dataset.num_classes).to(device)
    train(model, data)

    acc_before, loss_before = evaluate(model, data, data.test_mask)
    print(f"Accuracy before attack: {acc_before:.4f}")
    print(f"Loss before attack:     {loss_before:.4f}")

    target_nodes = select_target_nodes(data, strategy='high_degree', k=20)
    x_attacked = pgd_feature_attack(model, data, target_nodes)
    
    data_attacked = data.clone()
    data_attacked.x = x_attacked

    acc_after, loss_after = evaluate(model, data_attacked, data.test_mask)
    
    # Evaluate attack success on the target nodes
    pred_after = model(data_attacked.x, data_attacked.edge_index).argmax(dim=1)
    asr = (pred_after[target_nodes] != data.y[target_nodes]).float().mean().item()
    
    print(f"Accuracy after attack:  {acc_after:.4f}")
    print(f"Loss after attack:      {loss_after:.4f}")
    print(f"Attack Success Rate (ASR): {asr * 100:.2f}%")

if __name__ == '__main__':
    main()
