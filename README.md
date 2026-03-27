# Legal Multi-Label Classifier (BERT + GCN)

Quick start

1. Create a Python environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r legal_classifier/requirements.txt
```

2. Prepare data as JSONL files (one JSON object per line) with fields: `fact`, `accusation`, `relevant_articles`.

   - `train.json` (training)
   - `test.json` (validation)
   - `evaluation.json` (final evaluation)
3. Run training example:

```bash
python train.py \
  --train_path /root/autodl-tmp/dir_jcl/graduation_design/dataset/train_strict20.json \
  --valid_path /root/autodl-tmp/dir_jcl/graduation_design/dataset/test_strict20.json \
  --output_dir checkpoints \
  --epochs 5 \
  --batch_size 8
```

Notes

- The code uses `bert-base-chinese` by default. You can change the model in `train.py`.
- Labels (both accusations and law articles) are combined into a single multi-label space, with co-occurrence graph built from `train.json`.
