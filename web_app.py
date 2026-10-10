"""
LexMind AI - 司法文本智能分析系统
主应用入口 - Flask + SQLite + JWT认证
"""
import os
import json
from flask import Flask, request, jsonify, send_from_directory
from functools import wraps

# 设置Hugging Face模型下载路径（避免下载到C盘）
MODEL_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'external_models')
os.environ['HF_HOME'] = MODEL_CACHE_DIR
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

# 初始化数据库
from database import init_db, get_db
from auth import register_api as register_auth_routes
from blog_routes import register_blog_routes

app = Flask(__name__, static_folder='lexmind_frontend', static_url_path='')
app.config['SECRET_KEY'] = 'lexmind-legal-ai-secret-key-2024'

# CORS跨域支持
@app.after_request
def cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

# 处理OPTIONS预检请求
@app.route('/<path:path>', methods=['OPTIONS'])
@app.route('/api/<path:path>', methods=['OPTIONS'])
def options_handler(path=None):
    return '', 200

# 初始化数据库
init_db()

# 注册路由
register_auth_routes(app)
register_blog_routes(app)

# ==================== 预测API ====================

# 预测服务（懒加载）
prediction_service = None

def get_prediction_service():
    global prediction_service
    if prediction_service is None:
        from transformers import AutoTokenizer
        import torch
        torch.set_num_threads(os.cpu_count() or 4)
        from model import BertGCNForMultiLabel

        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), 'checkpoints', 'best_model.pt')
        BERT_MODEL = 'bert-base-chinese'
        MAX_LENGTH = 256

        if not os.path.exists(CHECKPOINT_PATH):
            raise FileNotFoundError(f"模型文件未找到: {CHECKPOINT_PATH}")

        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        label2idx = checkpoint.get('label2idx', {})
        idx2label = {v: k for k, v in label2idx.items()}
        num_labels = len(label2idx)

        model = BertGCNForMultiLabel(bert_model=BERT_MODEL, num_labels=num_labels)
        model.load_state_dict(checkpoint['model_state'])
        model.to(DEVICE)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)

        class PredictionService:
            def __init__(self):
                self.model = model
                self.tokenizer = tokenizer
                self.device = DEVICE
                self.idx2label = idx2label
                self.num_labels = num_labels
                self.max_length = MAX_LENGTH
                # 预计算identity矩阵，避免每次推理重建
                self._eye_cache = torch.eye(self.num_labels, device=self.device)

            def predict(self, facts, threshold=0.5):
                enc = self.tokenizer(
                    facts,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                input_ids = enc['input_ids'].to(self.device)
                attention_mask = enc['attention_mask'].to(self.device)
                adj = self._eye_cache

                with torch.inference_mode():
                    logits = self.model(input_ids=input_ids, attention_mask=attention_mask, adj=adj)
                    probs = torch.sigmoid(logits).cpu().numpy()

                max_conf = [0.0] * self.num_labels
                for row in probs:
                    for i, v in enumerate(row):
                        if v > max_conf[i]:
                            max_conf[i] = float(v)

                charges = []
                for idx, conf in enumerate(max_conf):
                    if conf >= threshold:
                        name = self.idx2label.get(idx, f'label_{idx}')
                        charges.append({'name': name, 'confidence': conf})

                return sorted(charges, key=lambda x: x['confidence'], reverse=True)

        prediction_service = PredictionService()

    return prediction_service

@app.route('/api/predict', methods=['POST'])
def predict():
    """罪名预测API"""
    try:
        data = request.get_json()
        facts = data.get('facts', [])
        threshold = float(data.get('threshold', 0.5))

        if not facts:
            return jsonify({'charges': [], 'articles': []})

        if isinstance(facts, str):
            facts = [facts]

        service = get_prediction_service()
        charges = service.predict(facts, threshold)

        # 保存预测记录（如果已登录）
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            from auth import decode_token
            from database import save_prediction, get_user_by_id
            token = auth_header[7:]
            user_id = decode_token(token)
            if user_id:
                user = get_user_by_id(user_id)
                if user:
                    input_text = '\n'.join(facts)
                    save_prediction(user_id, input_text, charges, threshold)

        return jsonify({'charges': charges, 'articles': []})

    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        return jsonify({'error': f'预测出错: {str(e)}'}), 500

@app.route('/api/model/status', methods=['GET'])
def model_status():
    """获取模型状态"""
    try:
        service = get_prediction_service()
        return jsonify({'status': 'ready', 'model': 'BERT-GCN'})
    except FileNotFoundError:
        return jsonify({'status': 'not_loaded', 'model': 'BERT-GCN'})

# ==================== 静态文件服务 ====================

@app.route('/')
def index():
    return send_from_directory('lexmind_frontend', 'index.html')

@app.route('/pages/<path:filename>')
def serve_pages(filename):
    return send_from_directory('lexmind_frontend/pages', filename)

@app.route('/api/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('lexmind_frontend', filename)

# ==================== 健康检查 ====================

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'LexMind AI'})


if __name__ == '__main__':
    print("=" * 50)
    print("LexMind AI - 司法文本智能分析系统")
    print("=" * 50)
    print("后端服务启动中...")
    print("访问 http://localhost:5000 使用系统")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=True)
