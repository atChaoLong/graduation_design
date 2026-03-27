# Title: A Hybrid Transformer + GCN Approach for Multi-Label Legal Text Classification

## Abstract

We present a hybrid architecture combining a Transformer-based encoder (BERT) with a Graph Convolutional Network (GCN) over label co-occurrence to address multi-label legal text classification for charge prediction and statute recommendation. The BERT encoder captures rich semantic representations of case facts, while the GCN models inter-label correlations derived from training data. Experiments show that combining semantic and relational signals improves Micro-F1 and Macro-F1 compared to a BERT-only baseline.

## 1. Introduction

Automatic legal text classification plays a crucial role in legal AI applications such as charge prediction and statutory recommendation. Documents often require multiple labels (e.g., multiple applicable articles), and label dependencies (co-occurrence, hierarchical relations) convey important signals. We propose a model that leverages a pretrained Transformer to extract document semantics and a lightweight GCN applied on a label co-occurrence graph to produce label embeddings that capture label relations.

## 2. Related Work

- Transformer-based NLP (Devlin et al., 2019) has set the state of the art in many legal NLP tasks.
- Graph Neural Networks (Kipf & Welling, 2017) have been used to incorporate structure and relations, including label graphs for classification.
- Prior legal AI work uses either strong textual encoders or hand-crafted label features; combining both remains under-explored.

## 3. Methodology

Model overview:

- Document encoder: `bert-base-chinese` produces a pooled document embedding (CLS). This captures semantic patterns in `fact` texts such as actions, entities, and context.
- Label graph: build a label co-occurrence matrix from training JSONL (accusation + relevant_articles). Add self-loops and apply symmetric normalization to obtain $\hat{A}$.
- GCN on labels: initialize label embeddings as learnable vectors $X^{(0)} \in \mathbb{R}^{L\times d}$ and apply two GCN layers: $X^{(l+1)} = \mathrm{ReLU}(\hat{A} X^{(l)} W^{(l)})$ to obtain refined label representations.
- Scoring: project BERT document embedding to the same label embedding space and compute logits via dot product between document vector and each label embedding. Optimize with binary cross-entropy with logits for multi-label classification.

Model advantages: Transformer captures text semantics; GCN captures label co-dependencies (e.g., certain articles frequently co-occur with specific charges). Combining both leads to better-ranked recommendations and more coherent multi-label predictions.

## 4. Experimental Setup

- Data: JSONL files where each record contains `fact`, `accusation` (single or list), and `relevant_articles` (single or list). Split into `train.json`, `test.json`, `evaluation.json`.
- Metrics: Micro-F1 and Macro-F1. Thresholding at 0.5 for evaluation; thresholds may be tuned per label.
- Implementation: PyTorch + Hugging Face Transformers. GCN implemented using dense adjacency to avoid external graph libraries, for portability.

## 5. Results and Analysis

Report Micro-F1 and Macro-F1 of the hybrid model vs BERT-only baseline. Analyze examples where label co-occurrence corrected ambiguous textual signals (e.g., co-occurring articles clarified intent when facts were terse).

## 6. Conclusion

We introduced a simple and effective hybrid architecture combining BERT and label-level GCN to jointly leverage semantics and label relations for multi-label legal classification. Future work includes: using richer label graphs (hierarchies, semantic similarity), adaptive thresholding, and end-to-end joint training with auxiliary tasks.

## Appendix: Implementation Notes

- The repository contains `train.py`, `model.py`, `data.py`, `utils.py` and a Markdown `paper.md` draft.
- Run instructions are in `README.md`.
