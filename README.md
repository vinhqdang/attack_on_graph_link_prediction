# attack_on_graph_link_prediction

Measuring different attacks on graph neural networks.

## Scripts

- `knowledge_graph_rgcn.py` – Example script that loads the AIFB knowledge graph dataset and trains an R-GCN with a simple adversarial edge/feature flipping attack.
- `topkusingpgd.py` - Example script that loads the Cora dataset and trains a GCN with a PGD-based feature attack.
- `GCN_vs_RGCN.ipynb` – Original notebook comparing a standard GCN with a noisy RGCN.

### Running the examples

Install the required packages:

```bash
pip install -r requirements.txt
```

Then, you can run the scripts:

```bash
python knowledge_graph_rgcn.py
python topkusingpgd.py
```

The scripts will print the model accuracy before and after the attack, as well as the attack success rate (ASR) and average modified links/features (AML).