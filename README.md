# graduation_design
我的毕业设计

# 前端启动
```bash
cd graduation_design
conda activate gd
python -m http.server 8001 --directory d:\work\my\graduation_design\lexmind_frontend
```

# 后端启动
```bash
cd graduation_design
conda activate gd
python web_app.py
```


### 对比实验

# TextCNN
```bash
python train_baseline.py --model textcnn --train_path dataset/train_strict20.json --valid_path dataset/evalution_strict20.json --epochs 10
```

# BiLSTM
```bash
python train_baseline.py --model bilstm --train_path dataset/train_strict20.json --valid_path dataset/evalution_strict20.json --epochs 10
```

# BERT baseline
```bash
python train_baseline.py --model bert --train_path dataset/train_strict20.json --valid_path dataset/evalution_strict20.json --epochs 10
```

# BERT+GCN
```bash
python train_baseline.py --model bert_gcn --train_path dataset/train_strict20.json --valid_path dataset/evalution_strict20.json --epochs 10
```

# BERT+GCN + Focal Loss
```bash
python train_baseline.py --model bert_gcn --loss focal --train_path dataset/train_strict20.json --valid_path dataset/evalution_strict20.json --epochs 10
```


### 消融实验

# 无GCN变体 - 直接用BERT baseline
```bash
python train_baseline.py --model bert --variant no_gcn ...
```

# 随机邻接矩阵
```bash
python train_baseline.py --model bert_gcn --variant random_adj ...
```

# 无归一化
```bash
python train_baseline.py --model bert_gcn --variant no_norm ...
```

# 对比模型
| 模型 | micro_f1 | macro_f1  | accuracy | loss function | optimizer | epochs |
| ---  | ---  | ---  | ---  | ---  | ---  | ---  |  
| bilstm | 0.9401 | 0.8387 | 0.9201 | bce | AdamW | 10 |
| textcnn | 0.9686 | 0.9321 | 0.9546 | bce | AdamW | 10 |
| bert_gcn+focal loss | 0.9760 | 0.9520 | 0.9710 | bce | AdamW | 10 |
# 消融实验
| 模型 | micro_f1 | macro_f1  | accuracy | loss function | optimizer | epochs |
| ---  | ---  | ---  | ---  | ---  | ---  | ---  |  
| bert | 0.9635 | 0.9106 | 0.9503 | bce | AdamW | 10 |
| bert_gcn | 0.9698 | 0.9393 | 0.9630  | bce | AdamW | 10 |
| bert_gcn+focal loss | 0.9760 | 0.9520 | 0.9710 | bce | AdamW | 10 |
数据集： CAIL 2018
数据集信息：
本次挑战赛所使用的数据集是来自“中国裁判文书网”公开的刑事法律文书，其中每份数据由法律文书中的案情描述和事实部分组成，同时也包括每个案件所涉及的法条、被告人被判的罪名和刑期长短等要素。
数据集共包括268万刑法法律文书，共涉及202条罪名，183条法条，刑期长短包括0-25年、无期、死刑。
我们将先后发布CAIL2018-Small和CAIL2018-Large两组数据集。CAIL2018-Small包括19.6万份文书样例，直接在该网站发布，包括15万训练集，1.6万验证集和3万测试集。CAIL2018-Large数据集，包括150万文书样例。剩余90万份文书将作为第一阶段的测试数据CAIL2018-Large-test。
使用该数据集请引用 CAIL2018: A Large-Scale Legal Dataset for Judgment Prediction 和 Overview of CAIL2018: Legal Judgment Prediction Competition。
仅过滤8种罪名作为使用，分别是：盗窃、危险驾驶、故意伤害、交通肇事、诈骗、寻衅滋事、抢劫、开设赌场