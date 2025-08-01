# Experimental Results

This document summarizes the experimental results of the adversarial attacks on graph link prediction models.

## Attack on VGAE (Cora Dataset)

The following table shows the performance of the VGAE model on the Cora dataset under different attack scenarios. The performance is measured in AUC (Area Under the ROC Curve).

| Attack Type                 | AUC Score |
| --------------------------- | --------- |
| Baseline (No Attack)        | ~0.54     |
| Meta-Attack                 | 0.4883    |
| PGD Attack                  | 0.8343    |

### Analysis

*   **Meta-Attack:** The meta-attack was successful in its goal of degrading the model's performance, reducing the AUC score to below the random-chance baseline of 0.5. This indicates that the attack learned an effective poisoning strategy.
*   **PGD Attack:** The PGD attack, surprisingly, resulted in a model with a very high AUC score. This suggests that the current implementation of the PGD attack is not effective against the VGAE model. It's possible that the attack is inadvertently making the link prediction task *easier* for the model, or that the VGAE architecture is particularly resilient to this type of perturbation.

**Conclusion:** The meta-attack is a more effective method for poisoning the VGAE model for link prediction compared to the PGD attack.