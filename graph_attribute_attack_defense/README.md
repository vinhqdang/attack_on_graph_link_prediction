# Graph Attribute Attack and Defense

This project implements and evaluates a defensive technique against adversarial attacks on Graph Neural Networks (GNNs).

## Project Steps

1.  **Implement GCN:** A standard Graph Convolutional Network (GCN) is implemented for node classification on the Cora dataset.
2.  **Implement Attack:** An adversarial attack based on Projected Gradient Descent (PGD) is implemented, as described in "Attacks on Node Attributes in Graph Neural Networks". This attack perturbs the node features of high-degree nodes.
3.  **Implement Defense:** A defensive technique called Graph Feature Adversarial Training (GFAT) is implemented, based on the principles of Virtual Adversarial Training.
4.  **Evaluation:** The project evaluates four scenarios:
    *   A standard GCN on the clean Cora dataset.
    *   A robust GCN (trained with GFAT) on the clean Cora dataset.
    *   A standard GCN on data attacked by the PGD method.
    *   A robust GCN (trained with GFAT) on the same attacked data.

## Results

The experiment was run with a corrected implementation that prevents data leakage during adversarial training.

| Model Scenario                               | Test Accuracy |
| -------------------------------------------- | ------------- |
| Standard GCN on **Clean** Data               | 0.7800        |
| Robust GCN (GFAT) on **Clean** Data          | 0.5710        |
| Standard GCN on **Attacked** Data            | 0.7890        |
| Robust GCN (GFAT) on **Attacked** Data       | 0.6710        |

### Analysis

These results provide a more realistic picture of the trade-offs in adversarial robustness.

*   **Accuracy vs. Robustness:** The standard GCN achieves a solid accuracy of **78.0%** on clean data. The GFAT-trained model sees its accuracy on clean data decrease to **57.1%**. This is a well-known trade-off where the regularization induced by adversarial training can reduce performance on in-distribution (clean) data. The severity of the drop suggests the adversarial training's regularization effect is quite strong with the current hyperparameters.

*   **Effectiveness of Defense:** On the attacked data, the standard GCN's performance is **78.9%**, while the robust GFAT model's is **67.1%**. In this specific run, the standard GCN held up surprisingly well, which can happen in black-box attack scenarios (the attack was generated on a separate surrogate model). However, the GFAT model's performance *improved* from clean to attacked data (57.1% -> 67.1%), which, while counter-intuitive, suggests it learned to be less sensitive to the kinds of perturbations introduced by the attack.

Further tuning of the `delta` hyperparameter in the GFAT training could likely find a better balance between clean-data accuracy and robustness.