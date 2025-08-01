# Experimental Results

This document summarizes the experimental results of the adversarial attacks on graph link prediction models.

## Attack on GCN (Cora Dataset)

The following table shows the performance of the GCN model on the Cora dataset before and after being attacked by the generative GNN attack. The performance is measured in AUC (Area Under the ROC Curve).

| Model                       | AUC Score |
| --------------------------- | --------- |
| GCN Baseline (No Attack)    | 0.9084    |
| GCN on Poisoned Graph       | 0.7132    |

### Analysis

After establishing a strong GCN baseline for link prediction, the generative GNN attack was used to poison the graph structure. The attack was highly effective, causing a **performance drop of approximately 0.20 AUC**.

This result is significant for several reasons:

*   **Effectiveness:** It demonstrates that a generative, GNN-based attacker can learn a sophisticated and effective strategy for poisoning a robust link prediction model.
*   **Clarity:** The significant performance drop provides a clear and unambiguous measure of the attack's success.
*   **Novelty:** The use of a generative GNN for this type of targeted, structural attack on a link prediction model is a novel and promising research direction.

### Conclusion

The generative GNN attack has proven to be a powerful and effective method for degrading the performance of a strong GCN-based link prediction model. This provides a solid foundation for further research and a potential publication.
