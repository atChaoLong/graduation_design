import os
from flask import Flask, request, render_template_string, jsonify
import torch
import numpy as np
from transformers import AutoTokenizer
from model import BertGCNForMultiLabel

app = Flask(__name__, static_folder='lexmind_frontend', static_url_path='')

# Config (can be overridden via env vars)
CHECKPOINT_PATH = r"D:\\work\\my\\graduation_design\\checkpoints\\best_model.pt"
BERT_MODEL = 'bert-base-chinese'
THRESHOLD = '0.5'
DEVICE = torch.device('cuda' if (torch.cuda.is_available() and os.environ.get('DEVICE','')!='cpu') else 'cpu')
MAX_LENGTH = int('256')

# load checkpo`int
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
<html lang="zh-CN">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>罪名&法条 识别</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="bg-light">
<div class="container py-5">
    <div class="card shadow-sm">
        <div class="card-body">
            <h3 class="card-title">法律事实识别</h3>
            <p class="text-muted">在下方文本框输入一条或多条事实，每条新行表示一个样本。系统将根据模型直接输出的最高置信度结果进行返回。</p>
            <form method="post" action="/predict">
                <div class="mb-3">
                    <textarea name="facts" class="form-control" rows="8" placeholder="每行一条事实"></textarea>
                </div>
                <button class="btn btn-primary">开始识别</button>
            </form>
        </div>
    </div>

    {% if results %}
    <div class="mt-4">
        <h4>预测结果</h4>
        <div class="row">
            {% for r in results %}
            <div class="col-md-6">
                <div class="card mb-3">
                    <div class="card-body">
                        <p class="card-text"><strong>Fact</strong>: {{r.fact}}</p>
                        <p class="card-text"><strong>Predicted Accusation</strong>: {{r.pred_accusation}}</p>
                        <p class="card-text"><strong>Predicted Article</strong>: {{r.pred_article}}</p>
                    </div>
                </div>
            </div>
            {% endfor %}
        </div>
    </div>
    {% endif %}
</div>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
'''


@app.route('/', methods=['GET'])
def index():
    # serve the frontend index (static)
    try:
        return app.send_static_file('index.html')
    except Exception:
        return render_template_string(HTML, results=None)


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Model not loaded. Place a valid checkpoint at {}".format(CHECKPOINT_PATH), 500

    facts_raw = request.form.get('facts','').strip()
    if not facts_raw:
        return render_template_string(HTML, results=[])

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

    # prepare label index groups
    acc_indices = [i for i, lab in idx2label.items() if lab.startswith('accusation::')]
    art_indices = [i for i, lab in idx2label.items() if lab.startswith('article::')]

    results = []
    for i, p in enumerate(probs):
        pred_acc = ''
        pred_art = ''
        # pick the highest-prob accusation (single-label decision)
        if acc_indices:
            acc_probs = p[acc_indices]
            acc_choice = int(acc_indices[np.argmax(acc_probs)])
            pred_acc = idx2label.get(acc_choice, '')
        # pick the highest-prob article (single best article)
        if art_indices:
            art_probs = p[art_indices]
            art_choice = int(art_indices[np.argmax(art_probs)])
            pred_art = idx2label.get(art_choice, '')

        results.append({'fact': facts[i], 'pred_accusation': pred_acc, 'pred_article': pred_art})

    return render_template_string(HTML, results=results)


# API endpoint for frontend to call with JSON
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json(force=True)
    facts = data.get('facts') if isinstance(data.get('facts'), list) else []
    threshold = float(data.get('threshold', THRESHOLD))

    if not facts:
        return jsonify({"charges": [], "articles": []})

    # If model is loaded, use it; otherwise fall back to simple keyword heuristic
    if model is not None and num_labels > 0:
        try:
            enc = tokenizer(facts, truncation=True, padding='max_length', max_length=MAX_LENGTH, return_tensors='pt')
            input_ids = enc['input_ids'].to(DEVICE)
            attention_mask = enc['attention_mask'].to(DEVICE)
            adj = torch.eye(num_labels, device=DEVICE)
            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask, adj=adj)
                probs = torch.sigmoid(logits).cpu().numpy()

            # aggregate confidences across samples by taking max per label
            max_conf = [0.0] * num_labels
            for row in probs:
                for i, v in enumerate(row):
                    if v > max_conf[i]:
                        max_conf[i] = float(v)

            charges = []
            articles = {}
            for idx, conf in enumerate(max_conf):
                if conf >= 0.0:
                    name = idx2label.get(idx, f'label_{idx}')
                    # if label encodes article as 'article::id::desc' try to parse
                    if isinstance(name, str) and name.startswith('article::'):
                        parts = name.split('::')
                        aid = parts[1] if len(parts) > 1 else parts[-1]
                        desc = parts[2] if len(parts) > 2 else ''
                        articles[aid] = {'id': aid, 'desc': desc}
                    charges.append({'name': name, 'confidence': conf})

            # sort charges by confidence desc and filter by threshold
            charges = sorted(charges, key=lambda x: x['confidence'], reverse=True)
            charges = [c for c in charges if c['confidence'] >= threshold]
            return jsonify({"charges": charges, "articles": list(articles.values())})
        except Exception as e:
            print('Error during model inference:', e)
            return jsonify({"error": str(e)}), 500
    else:
        # simple fallback heuristic (keyword-based)
        mapping = [
            (['偷','盗'], '盗窃罪', {'id':'264','desc':'盗窃罪相关法条'}),
            (['抢','抢劫'], '抢劫罪', {'id':'263','desc':'抢劫罪相关法条'}),
            (['诈骗','骗'], '诈骗罪', {'id':'266','desc':'诈骗罪相关法条'}),
            (['故意伤害','伤害'], '故意伤害罪', {'id':'234','desc':'故意伤害相关法条'}),
        ]
        charges_map = {}
        articles_map = {}
        import math
        for f in facts:
            lf = f.lower()
            for kws, name, art in mapping:
                for kw in kws:
                    if kw in lf:
                        prev = charges_map.get(name, 0.0)
                        # synthetic confidence
                        conf = min(0.99, 0.6 + min(0.35, lf.count(kw)*0.12) + min(0.25, math.log1p(len(f))/20))
                        charges_map[name] = max(prev, conf)
                        articles_map[art['id']] = art

        if not charges_map:
            # fallback pick
            name = '盗窃罪'
            charges_map[name] = 0.6

        charges = [{'name': k, 'confidence': v} for k, v in charges_map.items()]
        charges = sorted(charges, key=lambda x: x['confidence'], reverse=True)
        charges = [c for c in charges if c['confidence'] >= threshold]
        return jsonify({"charges": charges, "articles": list(articles_map.values())})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
