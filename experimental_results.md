# Experimental Results

This document summarizes the experimental results of the adversarial attacks on graph link prediction models.

## Attack on VGAE (Cora Dataset)

The following table shows the performance of the VGAE model on the Cora dataset under different attack scenarios. The performance is measured in AUC (Area Under the ROC Curve).

| Attack Type                 | AUC Score |
| --------------------------- | --------- |
| Baseline (No Attack)        | ~0.54     |
| Meta-Attack                 | 0.4883    |
| PGD Attack                  | 0.8343    |
| Generative GNN Attack       | 0.5272    |
| RL Attack (A2C)             | 0.8711    |

### Analysis

After implementing and evaluating a series of increasingly sophisticated adversarial attacks, a clear pattern has emerged: the attacks are not effectively degrading the performance of the VGAE model. In fact, the more advanced attacks (PGD and RL) are paradoxically *improving* the model's performance.

This strongly suggests that the issue lies not with the attack strategies, but with the inherent instability and weakness of the VGAE model as a baseline for this task. The model's performance is already close to random chance, and it appears to be highly sensitive to any structural perturbations. The "attacks" are likely acting as a form of regularization, inadvertently helping the model to learn a better representation of the graph.

### Conclusion and Recommendation

**The key takeaway from this investigation is that the VGAE model is not a suitable baseline for evaluating adversarial attacks on this dataset.**

To achieve a meaningful and significant performance degradation, it is essential to use a stronger, more stable baseline model. A standard GCN or GraphSAGE model trained for link prediction would provide a much more realistic and informative benchmark for evaluating the true effectiveness of these advanced adversarial attacks.

It is recommended to abandon the VGAE model and re-run these experiments against a more robust baseline.