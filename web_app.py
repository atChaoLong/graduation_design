import os
from flask import Flask, request, render_template_string, jsonify
import torch
import numpy as np
from transformers import AutoTokenizer
from model import BertGCNForMultiLabel

app = Flask(__name__)

# Config (can be overridden via env vars)
CHECKPOINT_PATH = os.environ.get('CHECKPOINT_PATH', 'checkpoints/best_model.pt')
BERT_MODEL = os.environ.get('BERT_MODEL', 'bert-base-chinese')
THRESHOLD = float(os.environ.get('THRESHOLD', '0.5'))
DEVICE = torch.device('cuda' if (torch.cuda.is_available() and os.environ.get('DEVICE','')!='cpu') else 'cpu')
MAX_LENGTH = int(os.environ.get('MAX_LENGTH', '256'))

# load checkpoint
if not os.path.exists(CHECKPOINT_PATH):
    print(f"Warning: checkpoint {CHECKPOINT_PATH} not found. Start server but predictions will fail until a checkpoint exists.")
    checkpoint = None
else:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')

# load label mapping
if checkpoint is not None and 'label2idx' in checkpoint:
    label2idx = checkpoint['label2idx']
else:
    label2idx = {}

idx2label = {v: k for k, v in label2idx.items()} if label2idx else {}
num_labels = len(label2idx)

# initialize model
if num_labels > 0:
    model = BertGCNForMultiLabel(bert_model=BERT_MODEL, num_labels=num_labels)
    if checkpoint is not None and 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    model.to(DEVICE)
    model.eval()
else:
    model = None

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

HTML = '''
<!doctype html>
<title>Legal Multi-label Inference</title>
<h1>输入事实（每行一条）</h1>
<form method=post action="/predict">
  <textarea name=facts rows=10 cols=80 placeholder="在此输入一条或多条事实，每条新行表示一个样本"></textarea><br>
  <label>概率阈值: <input name=threshold value="0.5"/></label>
  <input type=submit value=预测>
</form>
{% if results %}
<h2>预测结果</h2>
<ul>
{% for r in results %}
  <li>
    <b>Fact:</b> {{r.fact}}<br>
    <b>Predicted Labels:</b> {{r.pred_labels}}<br>
    <b>Predicted Articles:</b> {{r.pred_articles}}
  </li>
{% endfor %}
</ul>
{% endif %}
'''


@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML, results=None)


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Model not loaded. Place a valid checkpoint at {}".format(CHECKPOINT_PATH), 500

    facts_raw = request.form.get('facts','').strip()
    if not facts_raw:
        return render_template_string(HTML, results=[])

    threshold = float(request.form.get('threshold', THRESHOLD))
    facts = [f.strip() for f in facts_raw.splitlines() if f.strip()]

    # tokenize
    enc = tokenizer(facts, truncation=True, padding='max_length', max_length=MAX_LENGTH, return_tensors='pt')
    input_ids = enc['input_ids'].to(DEVICE)
    attention_mask = enc['attention_mask'].to(DEVICE)

    # build adjacency as identity (if you have cooccurrence matrix, you can build a better adj)
    adj = torch.eye(num_labels, device=DEVICE)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask, adj=adj)
        probs = torch.sigmoid(logits).cpu().numpy()

    results = []
    for i, p in enumerate(probs):
        preds = (p >= threshold).astype(int)
        pred_idx = np.where(preds == 1)[0].tolist()
        pred_labels = [idx2label[idx] for idx in pred_idx]
        pred_articles = [lab.split('::',1)[1] for lab in pred_labels if lab.startswith('article::')]
        results.append({'fact': facts[i], 'pred_labels': ','.join(pred_labels), 'pred_articles': ','.join(pred_articles)})

    return render_template_string(HTML, results=results)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
