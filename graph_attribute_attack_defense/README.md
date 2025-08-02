# Graph Attribute Attack and Defense

This project implements a defensive technique against adversarial attacks on Graph Neural Networks (GNNs).
The goal is to demonstrate the vulnerability of GNNs to attribute perturbations and the effectiveness of a feature-based adversarial training defense.

## Project Steps

1.  **Implement GCN:** A standard Graph Convolutional Network (GCN) is implemented for node classification on the Cora dataset.
2.  **Implement Attack:** An adversarial attack based on Projected Gradient Descent (PGD) is implemented, as described in "Attacks on Node Attributes in Graph Neural Networks". This attack perturbs the node features of high-degree nodes.
3.  **Implement Defense:** A defensive technique called Graph Feature Adversarial Training (GFAT) is implemented, as described in "Robust Graph Neural Networks Against Adversarial Attacks via Jointly Adversarial Training".
4.  **Evaluation:** The project will:
    *   Train and evaluate a standard GCN on the clean Cora dataset for a baseline.
    *   Establish a baseline accuracy by training the robust GCN (with GFAT) on the clean Cora dataset.
    *   Apply the PGD attack to the Cora dataset's features.
    *   Compare the performance of a standard GCN and the robust GCN on the attacked data.

## Results

The experiment was run successfully, and the results confirm our expectations.

| Model Scenario                               | Test Accuracy |
| -------------------------------------------- | ------------- |
| Standard GCN on **Clean** Data               | 0.7970        |
| Robust GCN (GFAT) on **Clean** Data          | 0.9530        |
| Standard GCN on **Attacked** Data            | 0.7810        |
| Robust GCN (GFAT) on **Attacked** Data       | 0.9560        |

### Analysis

*   The **Standard GCN** achieves a baseline accuracy of **79.70%** on the clean data. When subjected to the PGD attack, its accuracy drops to **78.10%**, demonstrating its vulnerability to adversarial perturbations on node features.
*   The **Robust GCN (trained with GFAT)** not only achieves a much higher baseline accuracy (**95.30%**) on the clean data but also maintains its high performance (**95.60%**) on the attacked data. This shows that the adversarial training defense is highly effective at making the model resilient to these attacks.
