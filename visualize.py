import os
import re
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

from data import build_label_space, build_label_cooccurrence, load_jsonl


def find_chinese_font():
    # try to find a font that supports Chinese glyphs
    candidates = [
        'Noto Sans CJK SC', 'NotoSansCJK', 'SimHei', 'WenQuanYi Zen Hei', 'WenQuanYi Micro Hei',
        'AR PL UKai CN', 'Microsoft YaHei', 'DejaVu Sans'
    ]
    sys_fonts = {Path(f).stem: f for f in font_manager.findSystemFonts()}
    for name in candidates:
        for fname in sys_fonts.values():
            if name.replace(' ', '').lower() in Path(fname).stem.replace('-', '').lower():
                return fname
    # fallback to DejaVu
    return font_manager.findfont('DejaVu Sans')


def parse_logs(logs_dir, start_time=None):
    logs_dir = Path(logs_dir)
    pattern = re.compile(r'(?P<ts>\d{14})_epoch(?P<ep>\d+)\.log')
    rows = []
    for p in sorted(logs_dir.glob('*.log')):
        m = pattern.match(p.name)
        if not m:
            continue
        ts = m.group('ts')
        ep = int(m.group('ep'))
        if start_time and ts != start_time:
            continue
        data = {'epoch': ep, 'path': str(p), 'timestamp': ts}
        # read file
        with open(p, 'r', encoding='utf-8') as f:
            for line in f:
                if '=' in line:
                    k, v = line.strip().split('=', 1)
                    try:
                        data[k] = float(v)
                    except Exception:
                        data[k] = v
        rows.append(data)
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values('epoch')
    return df


def load_predictions(pred_file):
    preds = []
    with open(pred_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                preds.append(json.loads(line))
    return preds


def plot_loss(df, out_path):
    sns.set(style='whitegrid')
    plt.figure(figsize=(6, 4), dpi=300)
    plt.plot(df['epoch'], df['loss'], marker='o', label='Train Loss')
    # smooth using rolling if many points
    if len(df) >= 3:
        plt.plot(df['epoch'], df['loss'].rolling(window=3, min_periods=1).mean(), linestyle='--', label='Smoothed')
    min_idx = df['loss'].idxmin()
    min_ep = int(df.loc[min_idx, 'epoch'])
    min_val = float(df.loc[min_idx, 'loss'])
    plt.scatter([min_ep], [min_val], color='red')
    plt.annotate(f'min: {min_val:.4f}', xy=(min_ep, min_val), xytext=(min_ep, min_val), textcoords='offset points')
    plt.title('Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_f1(df, out_path):
    sns.set(style='whitegrid')
    plt.figure(figsize=(6, 4), dpi=300)
    plt.plot(df['epoch'], df['micro_f1'], marker='o', linestyle='-', label='Micro-F1')
    plt.plot(df['epoch'], df['macro_f1'], marker='s', linestyle='--', label='Macro-F1')
    plt.title('F1 Scores')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_accuracy(df, out_path):
    sns.set(style='whitegrid')
    plt.figure(figsize=(6, 4), dpi=300)
    plt.plot(df['epoch'], df['accuracy'], marker='o', label='Exact Match Accuracy')
    plt.title('Exact Match Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_label_correlation(train_path, out_path, cmap='viridis', top_k=30):
    label2idx = build_label_space(train_path)
    cooc = build_label_cooccurrence(train_path, label2idx)
    # select top-k labels by degree
    deg = cooc.sum(axis=1)
    idx = np.argsort(-deg)[:top_k]
    sub = cooc[np.ix_(idx, idx)]
    labels = [list(label2idx.keys())[i] for i in idx]
    sns.set(style='whitegrid')
    plt.figure(figsize=(8, 6), dpi=300)
    sns.heatmap(sub, xticklabels=labels, yticklabels=labels, cmap=cmap)
    plt.title('Label Co-occurrence Heatmap')
    plt.xlabel('Label')
    plt.ylabel('Label')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_confusion(preds, label2idx, out_path, top_k=30):
    # build label list and map
    labels = list(label2idx.keys())
    L = len(labels)
    # build index mapping from label string to index
    idx_map = {lab: i for i, lab in enumerate(labels)}
    # construct co-occurrence confusion matrix: true x predicted
    M = np.zeros((L, L), dtype=np.int32)
    for rec in preds:
        true_keys = []
        for a in rec.get('orig_accusation', []) or []:
            k = f'accusation::{a}'
            if k in idx_map:
                true_keys.append(idx_map[k])
        for r in rec.get('orig_relevant_articles', []) or []:
            k = f'article::{r}'
            if k in idx_map:
                true_keys.append(idx_map[k])
        pred_keys = []
        for p in rec.get('predicted_labels', []) or []:
            if p in idx_map:
                pred_keys.append(idx_map[p])
        # increment counts
        for i in true_keys:
            for j in pred_keys:
                M[i, j] += 1

    # choose top_k labels by true support
    support = M.sum(axis=1)
    top_idx = np.argsort(-support)[:top_k]
    subM = M[np.ix_(top_idx, top_idx)]
    sub_labels = [labels[i] for i in top_idx]

    sns.set(style='whitegrid')
    plt.figure(figsize=(8, 6), dpi=300)
    sns.heatmap(subM, xticklabels=sub_labels, yticklabels=sub_labels, cmap='magma')
    plt.title('Label Confusion Matrix (true x predicted)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--logs_dir', type=str, default='checkpoints/logs')
    parser.add_argument('--pred_file', type=str, default=None)
    parser.add_argument('--start_time', type=str, default=None)
    parser.add_argument('--epoch', type=int, default=None)
    parser.add_argument('--output_dir', type=str, default='checkpoints/figures')
    parser.add_argument('--top_k', type=int, default=30)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # select logs dataframe
    df = parse_logs(args.logs_dir, start_time=args.start_time)
    if df is None or df.empty:
        raise RuntimeError(f'No logs found in {args.logs_dir}')

    # determine start_time and epoch to use
    if args.start_time is None:
        start_time = df['timestamp'].iloc[0]
    else:
        start_time = args.start_time
    if args.epoch is None:
        epoch = int(df['epoch'].max())
    else:
        epoch = args.epoch

    # filter df for start_time
    df_st = df[df['timestamp'] == start_time].sort_values('epoch')

    # set font for Chinese
    font_path = find_chinese_font()
    plt.rcParams['font.sans-serif'] = [font_path]
    plt.rcParams['axes.unicode_minus'] = False

    # plot loss
    loss_path = out_dir / f'{start_time}_epoch{epoch}_loss.png'
    plot_loss(df_st, loss_path)

    # plot f1
    f1_path = out_dir / f'{start_time}_epoch{epoch}_f1.png'
    plot_f1(df_st, f1_path)

    # plot accuracy
    acc_path = out_dir / f'{start_time}_epoch{epoch}_accuracy.png'
    plot_accuracy(df_st, acc_path)

    # label correlation heatmap
    corr_path = out_dir / f'{start_time}_epoch{epoch}_label_correlation.png'
    plot_label_correlation(args.train_path, corr_path, cmap='viridis', top_k=args.top_k)

    # predictions file
    pred_file = args.pred_file
    if pred_file is None:
        pred_file = Path(args.logs_dir) / f'{start_time}_epoch{epoch}_predictions.txt'
    if not Path(pred_file).exists():
        print(f'Prediction file {pred_file} not found; skipping confusion plot')
        return
    preds = load_predictions(pred_file)
    conf_path = out_dir / f'{start_time}_epoch{epoch}_confusion.png'
    label2idx = build_label_space(args.train_path)
    plot_confusion(preds, label2idx, conf_path, top_k=args.top_k)


if __name__ == '__main__':
    main()
import os
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from data import build_label_space, build_label_cooccurrence, load_jsonl


def setup_matplotlib_for_chinese():
    sns.set_style('whitegrid')
    # try common fonts for Chinese support
    fm = matplotlib.font_manager
    preferred = ['Noto Sans CJK JP', 'Noto Sans CJK SC', 'SimHei', 'WenQuanYi Zen Hei', 'DejaVu Sans']
    for name in preferred:
        try:
            matplotlib.rcParams['font.family'] = name
            _ = fm.findfont(name)
            return
        except Exception:
            continue
    # fallback: ensure sans-serif includes DejaVu
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'


def read_epoch_logs(logs_dir):
    logs = []
    for p in Path(logs_dir).glob('*_epoch*.log'):
        fname = p.name
        # filename format: {starttime}_epoch{n}.log
        parts = fname.split('_epoch')
        if len(parts) >= 2:
            start = parts[0]
            epoch_part = parts[1].split('.')[0]
            try:
                epoch = int(epoch_part)
            except Exception:
                continue
            d = {}
            with open(p, 'r', encoding='utf-8') as f:
                for line in f:
                    if '=' in line:
                        k, v = line.strip().split('=', 1)
                        d[k] = v
            d['start'] = start
            d['epoch'] = epoch
            logs.append(d)
    if not logs:
        return None, None
    df = pd.DataFrame(logs)
    df = df.sort_values('epoch')
    return df, df.iloc[-1]


def smooth(series, window=3):
    return series.astype(float).rolling(window=window, min_periods=1, center=True).mean()


def plot_loss(df, out_path, dpi=300):
    epochs = df['epoch'].astype(int)
    loss = df['loss'].astype(float)
    s_loss = smooth(loss, window=3)
    plt.figure(figsize=(6,4), dpi=dpi)
    plt.plot(epochs, s_loss, label='Train Loss', color='tab:blue')
    min_idx = s_loss.idxmin()
    min_epoch = int(df.loc[min_idx, 'epoch'])
    min_loss = float(s_loss.loc[min_idx])
    plt.scatter([min_epoch], [min_loss], color='red')
    plt.text(min_epoch, min_loss, f'  min={min_loss:.4f}\n  epoch={min_epoch}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train Loss Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def plot_f1(df, out_path, dpi=300):
    epochs = df['epoch'].astype(int)
    micro = df['micro_f1'].astype(float)
    macro = df['macro_f1'].astype(float)
    plt.figure(figsize=(6,4), dpi=dpi)
    plt.plot(epochs, micro, label='Micro-F1', color='tab:green', linestyle='-')
    plt.plot(epochs, macro, label='Macro-F1', color='tab:orange', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Micro-F1 and Macro-F1 over Epochs')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def plot_accuracy(df, out_path, dpi=300):
    epochs = df['epoch'].astype(int)
    acc = df['accuracy'].astype(float)
    plt.figure(figsize=(6,4), dpi=dpi)
    plt.plot(epochs, acc, label='Exact Match Accuracy', color='tab:purple')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Exact Match Accuracy over Epochs')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def plot_label_heatmap(cooc, labels, out_path, dpi=300, top_k=40, cmap='viridis'):
    # select top_k labels by degree
    degrees = cooc.sum(axis=1)
    idx = np.argsort(-degrees)[:top_k]
    sub = cooc[np.ix_(idx, idx)]
    sub_labels = [labels[i] for i in idx]
    plt.figure(figsize=(10,8), dpi=dpi)
    sns.heatmap(sub, xticklabels=sub_labels, yticklabels=sub_labels, cmap=cmap)
    plt.title('Label Correlation Heatmap')
    plt.xlabel('Label')
    plt.ylabel('Label')
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def plot_confusion_like(predictions_path, label2idx, out_path, dpi=300, top_k=40, cmap='magma'):
    # load predictions jsonl
    preds = []
    with open(predictions_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                preds.append(json.loads(line))
    # build label list
    idx2label = {v:k for k,v in label2idx.items()}
    labels = [idx2label[i] for i in range(len(idx2label))]
    L = len(labels)
    mat = np.zeros((L, L), dtype=int)
    for rec in preds:
        true_idxs = []
        for a in rec.get('orig_accusation', []) or []:
            key = f'accusation::{a}'
            if key in label2idx:
                true_idxs.append(label2idx[key])
        for r in rec.get('orig_relevant_articles', []) or []:
            key = f'article::{r}'
            if key in label2idx:
                true_idxs.append(label2idx[key])
        pred_labels = rec.get('predicted_labels', [])
        pred_idxs = [label2idx[l] for l in pred_labels if l in label2idx]
        for i in true_idxs:
            for j in pred_idxs:
                mat[i, j] += 1
    # select top_k by row sum
    row_sum = mat.sum(axis=1)
    idx = np.argsort(-row_sum)[:top_k]
    sub = mat[np.ix_(idx, idx)]
    sub_labels = [labels[i] for i in idx]
    plt.figure(figsize=(10,8), dpi=dpi)
    sns.heatmap(sub, xticklabels=sub_labels, yticklabels=sub_labels, cmap=cmap)
    plt.title('Label Co-occurrence: True (rows) vs Predicted (cols)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--logs_dir', type=str, default='checkpoints/logs')
    parser.add_argument('--out_dir', type=str, default='checkpoints/figures')
    parser.add_argument('--top_k', type=int, default=40)
    args = parser.parse_args()

    setup_matplotlib_for_chinese()

    logs_df, last = read_epoch_logs(args.logs_dir)
    if logs_df is None:
        print('No logs found in', args.logs_dir)
        return

    start_time = last['start']
    last_epoch = int(last['epoch'])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # convert columns to numeric where appropriate
    for col in ['loss','micro_f1','macro_f1','accuracy']:
        if col in logs_df.columns:
            logs_df[col] = pd.to_numeric(logs_df[col], errors='coerce')

    # Loss curve
    loss_path = out_dir / f'{start_time}_epoch{last_epoch}_loss.png'
    plot_loss(logs_df, loss_path)

    # F1 curves
    f1_path = out_dir / f'{start_time}_epoch{last_epoch}_f1.png'
    plot_f1(logs_df, f1_path)

    # Accuracy
    acc_path = out_dir / f'{start_time}_epoch{last_epoch}_accuracy.png'
    plot_accuracy(logs_df, acc_path)

    # build label co-occurrence and heatmap
    label2idx = build_label_space(args.train_path)
    cooc = build_label_cooccurrence(args.train_path, label2idx)
    heat_path = out_dir / f'{start_time}_epoch{last_epoch}_label_heatmap.png'
    plot_label_heatmap(cooc, list(label2idx.keys()), heat_path, top_k=args.top_k, cmap='viridis')

    # predictions file path
    pred_path = Path(args.logs_dir) / f'{start_time}_epoch{last_epoch}_predictions.txt'
    if pred_path.exists():
        conf_path = out_dir / f'{start_time}_epoch{last_epoch}_confusion.png'
        plot_confusion_like(pred_path, label2idx, conf_path, top_k=args.top_k)
    else:
        print('Predictions file not found:', pred_path)


if __name__ == '__main__':
    main()
