# Methodology

Our work introduces a novel poisoning attack framework, **Generative Adversarial Structure Perturbation (GASP)**, designed to effectively degrade the performance of Graph Neural Network (GNN) based models for link prediction. GASP frames the attack as a generative process, where a GNN-based attacker learns to produce a discrete, perturbed graph structure that maximizes the target model's prediction error. This approach overcomes key limitations of prior methods by directly optimizing for discrete structural changes in an end-to-end differentiable manner.

### 3.1. Problem Formulation

Let a graph be denoted as $G = (V, E)$, with a set of nodes $V$, edges $E$, and an associated feature matrix $X \in \mathbb{R}^{|V| \times d}$. The graph structure is represented by an adjacency matrix $A \in \{0, 1\}^{|V| \times |V|}$.

Our target is a GCN-based link prediction model, $f_\theta$, which learns a function to generate node embeddings $Z = f_\theta(X, A)$. The probability of a link between two nodes, $v_i$ and $v_j$, is then modeled by a decoder, such as the inner product of their embeddings: $p(A_{ij}=1 | Z) = \sigma(z_i^T z_j)$.

The goal of a poisoning attack is to learn a perturbation function $P$ that modifies the original adjacency matrix $A$ to a poisoned version $A' = P(A)$, subject to a perturbation budget $b$ (e.g., $||A' - A||_0 \le b$). The objective is to minimize the performance of a target model $f_{\theta'}$, trained from scratch on the poisoned graph $G' = (V, E', X)$, where $E'$ corresponds to $A'$.

### 3.2. The GASP Attacker

To address this challenge, we propose GASP, an attack model that learns to generate a highly effective, discrete perturbation matrix $A'$. The framework consists of two main components: a generative GNN attacker and a differentiable sampling mechanism.

#### 3.2.1. Generative Attacker Architecture

The core of GASP is a GNN-based attacker, $g_\phi$, which is designed to learn the optimal perturbation strategy. Unlike prior work that often learns a static perturbation matrix, our attacker is a generative model that takes the full graph context as input. The attacker is a multi-layer GCN that computes node representations sensitive to the attack objective:

$H_{atk} = g_\phi(X, A) = \text{GCN}_2(\text{ReLU}(\text{GCN}_1(X, A)))$

From these attack-aware embeddings, we generate a matrix of edge probabilities $P \in [0, 1]^{|V| \times |V|}$ by taking the inner product of the node embeddings, followed by a sigmoid activation:

$P = \sigma(H_{atk} \cdot H_{atk}^T)$

Each element $P_{ij}$ represents the learned probability that perturbing the edge $(v_i, v_j)$ will contribute to the attack's success.

#### 3.2.2. Differentiable Perturbation Sampling

A key challenge in generating adversarial graph structures is the discrete nature of the adjacency matrix, which makes direct gradient-based optimization intractable. To overcome this, we employ the **Gumbel-Softmax reparameterization trick**. This allows our model to make discrete, binary decisions (i.e., to keep or flip an edge) while maintaining a continuous, differentiable gradient flow back to the attacker's parameters $\phi$.

For each potential edge, we form a two-class probability distribution $[1 - P_{ij}, P_{ij}]$. We then sample from this distribution using the Gumbel-Softmax function to obtain a discrete, one-hot encoded decision for each edge. This yields a final, perturbed adjacency matrix $A'$ where edges have been stochastically added or removed based on the learned probabilities.

$A' = \text{GumbelSoftmax}([1-P, P], \tau)_{\text{hard=True}}$

The temperature parameter $\tau$ controls the smoothness of the approximation, which we anneal during training.

### 3.3. Training Objective

The attacker, $g_\phi$, is trained in a bi-level optimization loop. We first pre-train the target GCN model $f_\theta$ on the clean graph until convergence. Then, keeping the target model's parameters $\theta$ fixed, we train the attacker to generate perturbations that maximize the target's link prediction loss. The attacker's loss function is therefore the *negative* of the target's loss:

$\mathcal{L}_{atk}(\phi; \theta) = - \mathcal{L}_{\text{BCE}}(f_\theta(X, A'), Y_{\text{train}})$

where $A'$ is the perturbed adjacency matrix sampled from the attacker $g_\phi$, and $\mathcal{L}_{\text{BCE}}$ is the binary cross-entropy loss for the link prediction task. By minimizing $\mathcal{L}_{atk}$, the attacker learns to generate graph structures that are maximally confusing to the target GCN.

### 3.4. Comparison to Contemporary Approaches (2024-2025)

The GASP framework offers several advantages over existing and emerging adversarial attack methods:

1.  **vs. Gradient-Based Attacks (e.g., PGD):** While effective, PGD-based methods typically relax the discrete graph structure into a continuous domain to apply gradients. This can result in unrealistic perturbations and a mismatch between the optimized proxy and the final discrete graph. GASP, by contrast, directly generates discrete edge perturbations via the Gumbel-Softmax trick, ensuring a more realistic and potent attack.

2.  **vs. Meta-Learning Attacks:** Recent meta-learning approaches (as explored in our earlier experiments) often learn a static perturbation matrix. This is less powerful than our generative approach. GASP's GNN-based attacker can capture complex, high-order relationships in the graph to determine the most effective perturbations, learning a *function* that generates attacks rather than a single, fixed attack.

3.  **vs. Reinforcement Learning (RL) Attacks:** RL-based methods, while powerful, are known to suffer from high variance and instability during training, as demonstrated by our own experiments with A2C. GASP provides a more stable, end-to-end differentiable framework that avoids the complexities of reward design and policy gradient estimation, leading to more consistent and efficient training.

In summary, GASP advances the state-of-the-art by providing a stable, powerful, and end-to-end framework for generating discrete and highly effective adversarial attacks on graph structure.
