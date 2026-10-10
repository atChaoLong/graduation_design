"""
数据库模型 - SQLite
包含用户、文章、评论、预测记录等表
"""
import sqlite3
import os
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

DATABASE_PATH = 'legal_intelligence.db'

def get_db():
    """获取数据库连接"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """初始化数据库"""
    # 删除旧数据库重新创建（确保默认管理员账户正确）
    if os.path.exists(DATABASE_PATH):
        os.remove(DATABASE_PATH)
        print(f"删除旧数据库: {DATABASE_PATH}")

    conn = get_db()
    cursor = conn.cursor()

    # 创建用户表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            avatar_url TEXT,
            status INTEGER DEFAULT 1,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_login DATETIME
        )
    ''')

    # 创建分类表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            sort_order INTEGER DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 创建文章表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            summary TEXT,
            content TEXT NOT NULL,
            author_id INTEGER NOT NULL,
            category_id INTEGER,
            tags TEXT,
            view_count INTEGER DEFAULT 0,
            like_count INTEGER DEFAULT 0,
            comment_count INTEGER DEFAULT 0,
            status TEXT DEFAULT 'published',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (author_id) REFERENCES users(id),
            FOREIGN KEY (category_id) REFERENCES categories(id)
        )
    ''')

    # 创建评论表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            parent_id INTEGER,
            content TEXT NOT NULL,
            like_count INTEGER DEFAULT 0,
            status INTEGER DEFAULT 1,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (article_id) REFERENCES articles(id),
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (parent_id) REFERENCES comments(id)
        )
    ''')

    # 创建预测记录表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prediction_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            input_text TEXT NOT NULL,
            prediction_result TEXT NOT NULL,
            threshold REAL DEFAULT 0.5,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    # 创建文章点赞表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS article_likes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(article_id, user_id),
            FOREIGN KEY (article_id) REFERENCES articles(id),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    # 插入默认分类
    default_categories = [
        ('罪名解读', '深入解读各罪名的构成要件和认定标准', 1),
        ('案例分析', '真实案例分析和方法总结', 2),
        ('法律知识', '法律基础知识普及', 3),
        ('模型技术', '司法AI模型相关技术文章', 4),
    ]
    cursor.executemany(
        'INSERT OR IGNORE INTO categories (name, description, sort_order) VALUES (?, ?, ?)',
        default_categories
    )

    # 创建管理员账户 (admin / admin123)
    admin_salt = 'admin_salt_2024'
    admin_password = generate_password_hash('admin123' + admin_salt)
    cursor.execute('''
        INSERT OR IGNORE INTO users (username, email, password_hash, salt, role)
        VALUES (?, ?, ?, ?, ?)
    ''', ('admin', 'admin@lexmind.ai', admin_password, admin_salt, 'admin'))

    # 创建示例文章
    sample_articles = [
        (
            '盗窃罪的构成要件及司法实践',
            '本文深入分析盗窃罪的刑法规定、构成要件以及司法实践中的认定标准。',
            '''盗窃罪是日常生活中最常见的财产犯罪之一。根据《中华人民共和国刑法》第二百六十四条规定，盗窃公私财物，数额较大的，或者多次盗窃、入户盗窃、携带凶器盗窃、扒窃的，构成盗窃罪。

**盗窃罪的构成要件：**

1. **客体要件**：侵犯的是公私财物的所有权。
2. **客观要件**：实施了秘密窃取公私财物的行为。
3. **主体要件**：一般主体，年满16周岁。
4. **主观要件**：故意，并且具有非法占有目的。

**司法实践中的认定：**

在司法实践中，盗窃罪的认定需要考虑以下几个方面：
- 盗窃数额的认定（各地标准不同）
- 多次盗窃的认定（2年内3次以上）
- 入户盗窃的特殊认定
- 盗窃罪的量刑标准

通过AI模型进行盗窃罪预测时，需要关注案件事实中涉及的关键要素，如盗窃手段、涉案金额、盗窃地点等。''',
            1, 1, '盗窃罪,构成要件,司法实践'
        ),
        (
            '危险驾驶罪的法律分析与智能预测',
            '危险驾驶罪是2011年《刑法修正案（八）》增设的罪名，本文分析其法律适用。',
            '''危险驾驶罪是指在道路上驾驶机动车追逐竞驶，情节恶劣的，或者醉酒驾驶机动车的行为。

**危险驾驶罪的两种情形：**

1. **追逐竞驶**：指在道路上驾驶机动车追逐竞驶，情节恶劣的行为。
2. **醉酒驾驶**：指在道路上醉酒驾驶机动车的行为。

**法律依据：**

根据《刑法》第一百三十三条之一规定，危险驾驶罪的法定刑为拘役，并处罚金。

**醉驾标准：**

血液酒精含量达到80毫克/100毫升以上的，属于醉酒驾驶。

**AI预测要点：**

在预测危险驾驶罪时，需要关注：
- 是否存在醉酒驾驶情节
- 血液酒精含量
- 是否造成交通事故
- 是否有前科劣迹''',
            1, 1, '危险驾驶罪,醉驾,追逐竞驶'
        ),
        (
            '基于深度学习的司法文本罪名预测研究',
            '本文介绍如何使用BERT和图卷积网络进行刑事案件罪名预测。',
            '''**研究背景**

随着司法信息化的发展，利用人工智能技术辅助司法决策成为研究热点。罪名预测是法律文本智能分析的核心任务之一。

**技术方案**

本研究采用BERT+GCN的模型架构：

1. **BERT编码层**：利用预训练语言模型提取文本语义特征
2. **图卷积网络**：建模罪名标签之间的共现关系
3. **Focal Loss**：处理类别不平衡问题

**实验结果**

在CAIL 2018数据集上的实验表明：
- Micro-F1达到0.9760
- Macro-F1达到0.9520
- 准确率达到0.9710

**结论**

通过融合文本语义和标签关联信息，模型能够更准确地进行罪名预测。''',
            1, 4, 'BERT,GCN,罪名预测,深度学习'
        ),
    ]

    for article in sample_articles:
        cursor.execute('''
            INSERT OR IGNORE INTO articles (title, summary, content, author_id, category_id, tags)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', article)

    conn.commit()
    conn.close()
    print(f"数据库初始化完成: {DATABASE_PATH}")


# ==================== 用户操作 ====================

def create_user(username, email, password):
    """创建新用户"""
    conn = get_db()
    cursor = conn.cursor()

    salt = os.urandom(16).hex()
    password_hash = generate_password_hash(password + salt)

    try:
        cursor.execute('''
            INSERT INTO users (username, email, password_hash, salt)
            VALUES (?, ?, ?, ?)
        ''', (username, email, password_hash, salt))
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return {'id': user_id, 'username': username, 'email': email}
    except sqlite3.IntegrityError as e:
        conn.close()
        return None

def verify_user(email, password):
    """验证用户登录（通过邮箱）"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE email = ? AND status = 1', (email,))
    user = cursor.fetchone()
    conn.close()

    if user and check_password_hash(user['password_hash'], password + user['salt']):
        return dict(user)
    return None

def verify_user_by_login(login_input, password):
    """验证用户登录（支持用户名或邮箱）"""
    conn = get_db()
    cursor = conn.cursor()
    # 支持用户名或邮箱登录
    cursor.execute('SELECT * FROM users WHERE (username = ? OR email = ?) AND status = 1', (login_input, login_input))
    user = cursor.fetchone()
    conn.close()

    if user and check_password_hash(user['password_hash'], password + user['salt']):
        return dict(user)
    return None

def get_user_by_id(user_id):
    """根据ID获取用户"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT id, username, email, role, avatar_url, created_at FROM users WHERE id = ?', (user_id,))
    user = cursor.fetchone()
    conn.close()
    return dict(user) if user else None

def update_user_login_time(user_id):
    """更新最后登录时间"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?', (user_id,))
    conn.commit()
    conn.close()


# ==================== 文章操作 ====================

def create_article(title, content, author_id, summary='', category_id=None, tags=''):
    """创建文章"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO articles (title, summary, content, author_id, category_id, tags)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (title, summary, content, author_id, category_id, tags))
    conn.commit()
    article_id = cursor.lastrowid
    conn.close()
    return article_id

def get_articles(page=1, per_page=10, category_id=None, author_id=None, need_author=False):
    """获取文章列表

    Args:
        need_author: 是否联表查询作者名和分类名
    """
    conn = get_db()
    cursor = conn.cursor()

    if need_author:
        query = '''SELECT a.*, u.username as author_name, c.name as category_name
                   FROM articles a
                   LEFT JOIN users u ON a.author_id = u.id
                   LEFT JOIN categories c ON a.category_id = c.id
                   WHERE a.status = "published"'''
        count_query = 'SELECT COUNT(*) FROM articles WHERE status = "published"'
    else:
        query = 'SELECT * FROM articles WHERE status = "published"'
        count_query = 'SELECT COUNT(*) FROM articles WHERE status = "published"'
    params = []

    if category_id:
        query += ' AND category_id = ?'
        count_query += ' AND category_id = ?'
        params.append(category_id)

    if author_id:
        query += ' AND author_id = ?'
        count_query += ' AND author_id = ?'
        params.append(author_id)

    query += ' ORDER BY created_at DESC LIMIT ? OFFSET ?'
    params.extend([per_page, (page - 1) * per_page])

    # count_query 只需要 category_id 和 author_id 条件，不含 limit/offset
    count_params = [p for p in params if isinstance(p, int) and p != per_page and p != (page - 1) * per_page]
    # 重新构建 count 查询的参数
    count_params = []
    if category_id:
        count_params.append(category_id)
    if author_id:
        count_params.append(author_id)

    cursor.execute(count_query, count_params)
    total = cursor.fetchone()[0]

    cursor.execute(query, params)
    articles = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return articles, total

def get_article_by_id(article_id):
    """根据ID获取文章"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT a.*, u.username as author_name, c.name as category_name
        FROM articles a
        LEFT JOIN users u ON a.author_id = u.id
        LEFT JOIN categories c ON a.category_id = c.id
        WHERE a.id = ?
    ''', (article_id,))
    article = cursor.fetchone()
    conn.close()
    return dict(article) if article else None

def increment_article_view(article_id):
    """增加文章浏览量"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('UPDATE articles SET view_count = view_count + 1 WHERE id = ?', (article_id,))
    conn.commit()
    conn.close()

def get_categories():
    """获取所有分类"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM categories ORDER BY sort_order')
    categories = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return categories


# ==================== 评论操作 ====================

def add_comment(article_id, user_id, content, parent_id=None):
    """添加评论"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO comments (article_id, user_id, content, parent_id)
        VALUES (?, ?, ?, ?)
    ''', (article_id, user_id, content, parent_id))
    conn.commit()

    # 更新文章评论数
    cursor.execute('UPDATE articles SET comment_count = comment_count + 1 WHERE id = ?', (article_id,))
    conn.commit()
    conn.close()

def get_comments(article_id):
    """获取文章的评论"""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT c.*, u.username as user_name
        FROM comments c
        LEFT JOIN users u ON c.user_id = u.id
        WHERE c.article_id = ? AND c.status = 1
        ORDER BY c.created_at ASC
    ''', (article_id,))
    comments = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return comments


# ==================== 预测记录操作 ====================

def save_prediction(user_id, input_text, prediction_result, threshold=0.5):
    """保存预测记录"""
    conn = get_db()
    cursor = conn.cursor()
    import json
    cursor.execute('''
        INSERT INTO prediction_records (user_id, input_text, prediction_result, threshold)
        VALUES (?, ?, ?, ?)
    ''', (user_id, input_text, json.dumps(prediction_result, ensure_ascii=False), threshold))
    conn.commit()
    record_id = cursor.lastrowid
    conn.close()
    return record_id

def get_prediction_history(user_id, page=1, per_page=20):
    """获取用户的预测历史"""
    conn = get_db()
    cursor = conn.cursor()
    import json

    cursor.execute('''
        SELECT * FROM prediction_records
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT ? OFFSET ?
    ''', (user_id, per_page, (page - 1) * per_page))
    records = [dict(row) for row in cursor.fetchall()]

    # 解析JSON结果
    for record in records:
        record['prediction_result'] = json.loads(record['prediction_result'])

    cursor.execute('SELECT COUNT(*) FROM prediction_records WHERE user_id = ?', (user_id,))
    total = cursor.fetchone()[0]
    conn.close()

    return records, total


if __name__ == '__main__':
    init_db()
