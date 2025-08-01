# Experimental Results

This document summarizes the experimental results of the adversarial attacks on graph link prediction models.

## Meta-Attack on VGAE (Cora Dataset)

The following table shows the performance of the VGAE model on the Cora dataset before and after being attacked by the meta-learning based poisoning attack. The performance is measured in AUC (Area Under the ROC Curve).

| Model                       | AUC Score |
| --------------------------- | --------- |
| Baseline VGAE (Clean Graph) | ~0.54     |
| VGAE on Poisoned Graph      | 0.4883    |

**Conclusion:** The meta-attack successfully degraded the performance of the VGAE model, reducing the AUC score by approximately 0.05.
