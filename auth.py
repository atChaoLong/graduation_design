"""
用户认证模块 - JWT Token
"""
import datetime
import hmac
import hashlib
import base64
import json as json_lib
from functools import wraps
from flask import request, jsonify, current_app
from database import verify_user, create_user, get_user_by_id, update_user_login_time

SECRET_KEY = 'lexmind-legal-ai-secret-key-2024-lexmind'
TOKEN_EXPIRE_DAYS = 7

def generate_token(user_id):
    """生成简单Token"""
    payload = {
        'user_id': user_id,
        'exp': int((datetime.datetime.utcnow() + datetime.timedelta(days=TOKEN_EXPIRE_DAYS)).timestamp()),
        'iat': int(datetime.datetime.utcnow().timestamp())
    }
    payload_str = json_lib.dumps(payload)
    payload_b64 = base64.urlsafe_b64encode(payload_str.encode()).decode()

    sig = hmac.new(SECRET_KEY.encode(), payload_b64.encode(), hashlib.sha256).digest()
    sig_b64 = base64.urlsafe_b64encode(sig).decode()

    return f"{payload_b64}.{sig_b64}"

def decode_token(token):
    """解码Token"""
    try:
        parts = token.split('.')
        if len(parts) != 2:
            return None

        payload_b64, sig_b64 = parts

        # 验证签名
        expected_sig = hmac.new(SECRET_KEY.encode(), payload_b64.encode(), hashlib.sha256).digest()
        expected_sig_b64 = base64.urlsafe_b64encode(expected_sig).decode()

        if not hmac.compare_digest(sig_b64, expected_sig_b64):
            return None

        # 解析payload
        payload_str = base64.urlsafe_b64decode(payload_b64.encode()).decode()
        payload = json_lib.loads(payload_str)

        # 检查过期
        if payload.get('exp', 0) < int(datetime.datetime.utcnow().timestamp()):
            return None

        return payload.get('user_id')
    except Exception:
        return None

def require_auth(f):
    """登录验证装饰器"""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None

        # 从请求头获取Token
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header[7:]

        if not token:
            return jsonify({'error': '未提供认证Token'}), 401

        user_id = decode_token(token)
        if not user_id:
            return jsonify({'error': 'Token无效或已过期'}), 401

        user = get_user_by_id(user_id)
        if not user:
            return jsonify({'error': '用户不存在'}), 401

        # 将用户信息传递给视图函数
        return f(user, *args, **kwargs)

    return decorated

def require_admin(f):
    """管理员验证装饰器"""
    @wraps(f)
    @require_auth
    def decorated(user, *args, **kwargs):
        if user['role'] != 'admin':
            return jsonify({'error': '需要管理员权限'}), 403
        return f(user, *args, **kwargs)
    return decorated

# ==================== 认证API路由 ====================

def register_api(app):
    """注册用户相关API"""
    @app.route('/api/register', methods=['POST'])
    def register():
        data = request.get_json()
        username = data.get('username', '').strip()
        email = data.get('email', '').strip()
        password = data.get('password', '')

        # 验证输入
        if not username or not email or not password:
            return jsonify({'error': '请填写所有必填字段'}), 400

        if len(password) < 6:
            return jsonify({'error': '密码长度至少6位'}), 400

        if '@' not in email:
            return jsonify({'error': '请输入有效的邮箱地址'}), 400

        user = create_user(username, email, password)
        if not user:
            return jsonify({'error': '用户名或邮箱已被注册'}), 409

        return jsonify({
            'message': '注册成功',
            'user': user
        }), 201

    @app.route('/api/login', methods=['POST'])
    def login():
        data = request.get_json()
        login_input = data.get('email', '').strip()  # 支持用户名或邮箱
        password = data.get('password', '')

        if not login_input or not password:
            return jsonify({'error': '请输入用户名/邮箱和密码'}), 400

        # 支持用户名或邮箱登录
        from database import verify_user_by_login
        user = verify_user_by_login(login_input, password)
        if not user:
            return jsonify({'error': '用户名/邮箱或密码错误'}), 401

        # 更新登录时间
        update_user_login_time(user['id'])

        # 生成Token
        token = generate_token(user['id'])

        return jsonify({
            'message': '登录成功',
            'token': token,
            'user': {
                'id': user['id'],
                'username': user['username'],
                'email': user['email'],
                'role': user['role']
            }
        })

    @app.route('/api/user/info', methods=['GET'])
    @require_auth
    def get_user_info(user):
        return jsonify({'user': user})

    @app.route('/api/user/history', methods=['GET'])
    @require_auth
    def get_user_history(user):
        from database import get_prediction_history
        page = request.args.get('page', 1, type=int)
        records, total = get_prediction_history(user['id'], page)
        return jsonify({
            'records': records,
            'total': total,
            'page': page
        })
