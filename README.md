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
micro_f1=0.975989
macro_f1=0.125702
accuracy=0.958454


# BiLSTM
```bash
python train_baseline.py --model bilstm --train_path dataset/train_strict20.json --valid_path dataset/evalution_strict20.json --epochs 10
```

# BERT baseline
```bash
python train_baseline.py --model bert --train_path dataset/train_strict20.json --valid_path dataset/evalution_strict20.json --epochs 10
```

# BERT+GCN (你的模型)
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