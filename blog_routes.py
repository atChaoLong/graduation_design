"""
博客系统API路由
"""
from flask import request, jsonify
from auth import require_auth, require_admin
from database import (
    get_articles, get_article_by_id, create_article,
    get_categories, add_comment, get_comments, increment_article_view
)

def register_blog_routes(app):
    """注册博客相关API"""

    # 获取分类列表
    @app.route('/api/categories', methods=['GET'])
    def list_categories():
        categories = get_categories()
        return jsonify({'categories': categories})

    # 获取文章列表
    @app.route('/api/articles', methods=['GET'])
    def list_articles():
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        category_id = request.args.get('category_id', type=int)

        articles, total = get_articles(page, per_page, category_id, need_author=True)
        return jsonify({
            'articles': articles,
            'total': total,
            'page': page,
            'pages': (total + per_page - 1) // per_page
        })

    # 获取我的文章
    @app.route('/api/my/articles', methods=['GET'])
    @require_auth
    def my_articles(user):
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        articles, total = get_articles(page, per_page, author_id=user['id'], need_author=True)
        return jsonify({
            'articles': articles,
            'total': total,
            'page': page
        })

    # 获取文章详情
    @app.route('/api/articles/<int:article_id>', methods=['GET'])
    def get_article(article_id):
        article = get_article_by_id(article_id)
        if not article:
            return jsonify({'error': '文章不存在'}), 404

        # 增加浏览量
        increment_article_view(article_id)

        # 获取评论
        comments = get_comments(article_id)

        return jsonify({
            'article': article,
            'comments': comments
        })

    # 创建文章
    @app.route('/api/articles', methods=['POST'])
    @require_auth
    def new_article(user):
        data = request.get_json()
        title = data.get('title', '').strip()
        content = data.get('content', '').strip()
        summary = data.get('summary', '').strip()
        category_id = data.get('category_id')
        if category_id is not None:
            category_id = int(category_id)
        tags = data.get('tags', '').strip()

        if not title or not content:
            return jsonify({'error': '标题和内容不能为空'}), 400

        article_id = create_article(
            title=title,
            content=content,
            author_id=user['id'],
            summary=summary,
            category_id=category_id,
            tags=tags
        )

        return jsonify({
            'message': '文章发布成功',
            'article_id': article_id
        }), 201

    # 更新文章
    @app.route('/api/articles/<int:article_id>', methods=['PUT'])
    @require_auth
    def update_article(user, article_id):
        article = get_article_by_id(article_id)
        if not article:
            return jsonify({'error': '文章不存在'}), 404

        if article['author_id'] != user['id'] and user['role'] != 'admin':
            return jsonify({'error': '无权限修改'}), 403

        data = request.get_json()

        # 更新数据库
        from database import get_db
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE articles
            SET title = ?, summary = ?, content = ?, category_id = ?, tags = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (
            data.get('title', article['title']),
            data.get('summary', article['summary']),
            data.get('content', article['content']),
            data.get('category_id', article['category_id']),
            data.get('tags', article['tags']),
            article_id
        ))
        conn.commit()
        conn.close()

        return jsonify({'message': '更新成功'})

    # 删除文章
    @app.route('/api/articles/<int:article_id>', methods=['DELETE'])
    @require_auth
    def delete_article(user, article_id):
        article = get_article_by_id(article_id)
        if not article:
            return jsonify({'error': '文章不存在'}), 404

        if article['author_id'] != user['id'] and user['role'] != 'admin':
            return jsonify({'error': '无权限删除'}), 403

        from database import get_db
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM articles WHERE id = ?', (article_id,))
        conn.commit()
        conn.close()

        return jsonify({'message': '删除成功'})

    # 添加评论
    @app.route('/api/articles/<int:article_id>/comments', methods=['POST'])
    @require_auth
    def new_comment(user, article_id):
        article = get_article_by_id(article_id)
        if not article:
            return jsonify({'error': '文章不存在'}), 404

        data = request.get_json()
        content = data.get('content', '').strip()
        parent_id = data.get('parent_id', type=int)

        if not content:
            return jsonify({'error': '评论内容不能为空'}), 400

        add_comment(article_id, user['id'], content, parent_id)

        return jsonify({'message': '评论成功'}), 201

    # 获取评论列表
    @app.route('/api/articles/<int:article_id>/comments', methods=['GET'])
    def article_comments(article_id):
        comments = get_comments(article_id)
        return jsonify({'comments': comments})
