# attack_on_graph_link_prediction

Measuring different attacks on graph neural networks.

## Scripts

- `GCN_vs_RGCN.ipynb` – Original notebook comparing a standard GCN with a noisy RGCN.
- `knowledge_graph_rgcn.py` – Example script that loads the AIFB knowledge graph dataset and trains an R-GCN with a simple adversarial edge/feature flipping attack.

### Running the knowledge graph example

Install the required packages (PyTorch and PyTorch Geometric) and then execute:

```bash
python knowledge_graph_rgcn.py
```

The script will print the model accuracy before and after the attack as well as the attack success rate (ASR) and average modified links (AML).
