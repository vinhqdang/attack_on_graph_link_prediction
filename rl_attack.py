import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, VGAE
from torch_geometric.utils import dense_to_sparse, to_dense_adj, train_test_split_edges
from sklearn.metrics import roc_auc_score, average_precision_score

# Define the GNN Agent for the RL attack
class RLAgent(nn.Module):
    def __init__(self, n_feat, n_hid=32):
        super(RLAgent, self).__init__()
        self.gcn1 = GCNConv(n_feat, n_hid)
        self.gcn2 = GCNConv(n_hid, n_hid)
        self.gcn3 = GCNConv(n_hid, n_hid)
        
        # Actor head
        self.actor_head = nn.Linear(n_hid, 1)
        
        # Critic head
        self.critic_head = nn.Linear(n_hid, 1)

    def forward(self, x, adj):
        x = F.relu(self.gcn1(x, adj))
        x = F.relu(self.gcn2(x, adj))
        x_actor = F.relu(self.gcn3(x, adj))
        x_critic = F.relu(self.gcn3(x, adj))
        
        # Generate action probabilities
        action_scores = torch.sigmoid(torch.matmul(x_actor, x_actor.t()))
        
        # Generate state value
        state_values = self.critic_head(x_critic)
        
        return action_scores.view(-1), torch.mean(state_values)

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
    
    # --- Pre-train a target VGAE model ---
    print("--- Pre-training Target VGAE Model ---")
    target_vgae = CustomVGAE(VariationalGCNEncoder(n_features, out_channels)).to(device)
    optimizer_vgae = torch.optim.Adam(target_vgae.parameters(), lr=0.01)
    for epoch in range(200):
        target_vgae.train()
        optimizer_vgae.zero_grad()
        z = target_vgae.encode(data.x, data.train_pos_edge_index)
        loss = target_vgae.recon_loss(z, data.train_pos_edge_index) + (1 / n_nodes) * target_vgae.kl_loss()
        loss.backward()
        optimizer_vgae.step()
    target_vgae.eval()

    # --- Train the RL Agent ---
    print("\n--- Training RL Agent ---")
    agent = RLAgent(n_features).to(device)
    optimizer_agent = torch.optim.Adam(agent.parameters(), lr=0.001)
    adj_orig = to_dense_adj(data.train_pos_edge_index, max_num_nodes=n_nodes)[0]
    budget = 50 # Number of edges to flip

    for epoch in range(50): # 50 episodes
        agent.train()
        optimizer_agent.zero_grad()

        # Get action probabilities and state value from the agent
        action_probs, state_value = agent(data.x, data.train_pos_edge_index)
        
        # Sample actions (edges to flip)
        dist = Categorical(action_probs)
        actions = dist.sample((budget,))
        
        # Create the poisoned graph
        poisoned_adj = adj_orig.clone()
        for action in actions:
            row = action // n_nodes
            col = action % n_nodes
            poisoned_adj[row, col] = 1 - poisoned_adj[row, col] # Flip the edge
            poisoned_adj[col, row] = 1 - poisoned_adj[col, row] # Ensure symmetry
        
        poisoned_edge_index, _ = dense_to_sparse(poisoned_adj)

        # Get the reward
        with torch.no_grad():
            z = target_vgae.encode(data.x, poisoned_edge_index)
            auc, _ = target_vgae.test(z, data.test_pos_edge_index, data.test_neg_edge_index)
            reward = -auc # We want to minimize the AUC

        # Calculate the advantage
        advantage = reward - state_value.detach()

        # Calculate the policy loss (actor loss)
        log_probs = dist.log_prob(actions)
        actor_loss = -torch.mean(log_probs * advantage)
        
        # Calculate the value loss (critic loss)
        critic_loss = F.mse_loss(state_value, torch.tensor(reward, device=device, dtype=state_value.dtype))
        
        # Total loss
        loss = actor_loss + critic_loss
        
        loss.backward()
        optimizer_agent.step()

        if epoch % 5 == 0:
            print(f'Epoch: {epoch:03d}, Reward (AUC): {-reward:.4f}, Loss: {loss.item():.4f}')

    print("--- RL Agent Training Finished ---")

    # --- Evaluation Phase ---
    print("\n--- Evaluating Attack Performance ---")
    agent.eval()
    action_probs, _ = agent(data.x, data.train_pos_edge_index)
    _, top_actions = torch.topk(action_probs, budget) # Select the best actions
    
    final_poisoned_adj = adj_orig.clone()
    for action in top_actions:
        row = action // n_nodes
        col = action % n_nodes
        final_poisoned_adj[row, col] = 1 - final_poisoned_adj[row, col]
        final_poisoned_adj[col, row] = 1 - final_poisoned_adj[col, row]
        
    final_poisoned_edge_index, _ = dense_to_sparse(final_poisoned_adj)

    print("Training VGAE on the POISONED graph...")
    poisoned_vgae = CustomVGAE(VariationalGCNEncoder(n_features, out_channels)).to(device)
    optimizer_poisoned = torch.optim.Adam(poisoned_vgae.parameters(), lr=0.01)

    for epoch in range(200):
        poisoned_vgae.train()
        optimizer_poisoned.zero_grad()
        z = poisoned_vgae.encode(data.x, final_poisoned_edge_index)
        loss = poisoned_vgae.recon_loss(z, final_poisoned_edge_index) + (1 / n_nodes) * poisoned_vgae.kl_loss()
        loss.backward()
        optimizer_poisoned.step()

    poisoned_vgae.eval()
    with torch.no_grad():
        z = poisoned_vgae.encode(data.x, final_poisoned_edge_index)
        final_auc, _ = poisoned_vgae.test(z, data.test_pos_edge_index, data.test_neg_edge_index)

    print("\n--- RL Attack Results ---")
    print(f"VGAE on RL Poisoned Graph AUC: {final_auc:.4f}")

if __name__ == '__main__':
    main()
