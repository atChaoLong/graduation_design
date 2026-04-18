import os
import argparse
import logging
from pathlib import Path
import numpy as np
import json
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from tqdm import tqdm
 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix
from data import LegalJSONLDataset, build_label_space, build_label_cooccurrence
from model import BertGCNForMultiLabel
from utils import normalize_adj, save_checkpoint
from sklearn.metrics import f1_score


def compute_metrics(y_true, y_pred):
    # y_true, y_pred are numpy arrays (N, L)
    micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    # sample-wise exact match accuracy
    if y_true.shape[0] == 0:
        acc = 0.0
    else:
        acc = (y_true == y_pred).all(axis=1).mean()
    return micro, macro, acc


def plot_confusion_matrix(confusion, classes, title, save_path):
    """Plot and save a confusion matrix with the style from the user's example."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.cpu().numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticks(np.arange(len(classes)), labels=classes, rotation=90)
    ax.set_yticks(np.arange(len(classes)), labels=classes)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)

    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def build_label_confusion_matrices(y_true, y_pred, acc_indices, article_indices, acc_classes, article_classes):
    """
    Build confusion matrices for accusations and articles separately.
    For multi-label: each sample can contribute to multiple cells (one per true label and one per predicted label).
    We normalize by row (true label distribution).
    Returns (acc_confusion, article_confusion)
    """
    import torch
    # accusations confusion matrix
    acc_confusion = torch.zeros(len(acc_classes), len(acc_classes))
    # articles confusion matrix
    article_confusion = torch.zeros(len(article_classes), len(article_classes))

    # Extract only accusation columns and article columns
    y_true_acc = y_true[:, acc_indices]
    y_pred_acc = y_pred[:, acc_indices]
    y_true_art = y_true[:, article_indices]
    y_pred_art = y_pred[:, article_indices]

    # For each sample, add counts
    for i in range(y_true.shape[0]):
        # accusations: for each true=1 and pred=1, increment confusion[t][p]
        true_acc_idx = np.where(y_true_acc[i] == 1)[0]
        pred_acc_idx = np.where(y_pred_acc[i] == 1)[0]
        for t in true_acc_idx:
            for p in pred_acc_idx:
                acc_confusion[t][p] += 1

        # articles
        true_art_idx = np.where(y_true_art[i] == 1)[0]
        pred_art_idx = np.where(y_pred_art[i] == 1)[0]
        for t in true_art_idx:
            for p in pred_art_idx:
                article_confusion[t][p] += 1

    # Normalize by dividing every row by its sum
    for i in range(len(acc_classes)):
        denom = acc_confusion[i].sum()
        if denom > 0:
            acc_confusion[i] = acc_confusion[i] / denom

    for i in range(len(article_classes)):
        denom = article_confusion[i].sum()
        if denom > 0:
            article_confusion[i] = article_confusion[i] / denom

    return acc_confusion, article_confusion


def collate_fn(batch):
    input_ids = torch.stack([b['input_ids'] for b in batch])
    attention_mask = torch.stack([b['attention_mask'] for b in batch])
    # 修复：先转 numpy array 再转 torch tensor
    labels_np = np.array([b['labels'] for b in batch])
    labels = torch.from_numpy(labels_np).to(torch.float32)
    # collect raw fields (lists) for downstream validation output
    raw_facts = [b.get('raw_fact', '') for b in batch]
    orig_accs = [b.get('orig_accusation', []) for b in batch]
    orig_rels = [b.get('orig_relevant_articles', []) for b in batch]
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'raw_facts': raw_facts,
        'orig_accs': orig_accs,
        'orig_rels': orig_rels,
    }


def train(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('train')

    label2idx = build_label_space(args.train_path)
    num_labels = len(label2idx)
    logger.info(f'Number of labels: {num_labels}')

    # build co-occurrence and adjacency
    cooc = build_label_cooccurrence(args.train_path, label2idx)
    adj = normalize_adj(cooc)
    adj = torch.tensor(adj, dtype=torch.float32)

    train_ds = LegalJSONLDataset(args.train_path, label2idx, max_length=args.max_length)
    valid_ds = LegalJSONLDataset(args.valid_path, label2idx, max_length=args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Extract accusation and article indices and class names
    acc_indices = []
    article_indices = []
    acc_classes = []
    article_classes = []
    for label, idx in label2idx.items():
        if label.startswith('accusation::'):
            acc_indices.append(idx)
            acc_classes.append(label.split('::', 1)[1])
        elif label.startswith('article::'):
            article_indices.append(idx)
            article_classes.append(label.split('::', 1)[1])
    # Sort by index to maintain consistent order
    acc_order = np.argsort(acc_indices)
    acc_indices = [acc_indices[i] for i in acc_order]
    acc_classes = [acc_classes[i] for i in acc_order]
    article_order = np.argsort(article_indices)
    article_indices = [article_indices[i] for i in article_order]
    article_classes = [article_classes[i] for i in article_order]
    logger.info(f'Number of accusation classes: {len(acc_classes)}')
    logger.info(f'Accusation classes: {acc_classes}')
    logger.info(f'Number of article classes: {len(article_classes)}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertGCNForMultiLabel(bert_model=args.bert_model, num_labels=num_labels, label_dim=args.label_dim)
    model.to(device)
    adj = adj.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    criterion = BCEWithLogitsLoss()

    best_micro = 0.0
    start_time = datetime.now().strftime('%Y%m%d%H%M%S')

    logs_root = Path(args.output_dir) / 'logs'
    logs_root.mkdir(parents=True, exist_ok=True)
    run_dir = logs_root / start_time
    run_dir.mkdir(parents=True, exist_ok=True)

    # lists for plotting
    epoch_losses = []
    epoch_micro_f1 = []
    epoch_macro_f1 = []
    epoch_accs = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        processed = 0
        correct = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)
        for batch in pbar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask, adj=adj)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * input_ids.size(0)
            # update running accuracy (sample-wise exact match)
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs >= args.threshold).int().cpu().numpy()
                labs = labels.cpu().numpy().astype(int)
                batch_size = labs.shape[0]
                processed += batch_size
                correct += int((labs == preds).all(axis=1).sum())
                train_acc = correct / processed if processed > 0 else 0.0
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'train_acc': f'{train_acc:.4f}'})

        epoch_loss = running_loss / len(train_ds)

        # validation
        model.eval()
        y_true = []
        y_pred = []
        # prepare index->label mapping for readable outputs
        idx2label = {v: k for k, v in label2idx.items()}
        pred_file_path = run_dir / f'epoch_{epoch}.txt'
        with open(pred_file_path, 'w', encoding='utf-8') as pf:
            with torch.no_grad():
                for batch in tqdm(valid_loader, desc='Valid', leave=False):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].cpu().numpy()
                    logits = model(input_ids=input_ids, attention_mask=attention_mask, adj=adj)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    preds = (probs >= args.threshold).astype(int)
                    y_true.append(labels)
                    y_pred.append(preds)

                    # write per-sample readable outputs
                    raw_facts = batch.get('raw_facts', [])
                    orig_accs = batch.get('orig_accs', [])
                    orig_rels = batch.get('orig_rels', [])
                    for i in range(preds.shape[0]):
                        pred_idx = list(np.where(preds[i] == 1)[0])
                        pred_labels = [idx2label[idx] for idx in pred_idx]
                        pred_accs = [lbl.split('::', 1)[1] for lbl in pred_labels if lbl.startswith('accusation::')]
                        pred_articles = [lbl.split('::', 1)[1] for lbl in pred_labels if lbl.startswith('article::')]
                        fact_line = raw_facts[i] if i < len(raw_facts) else ''
                        orig_acc_line = ','.join(orig_accs[i]) if i < len(orig_accs) else ''
                        orig_rel_line = ','.join(orig_rels[i]) if i < len(orig_rels) else ''
                        pred_acc_line = ','.join(pred_accs)
                        pred_rel_line = ','.join(pred_articles)
                        # write three-line format per user request
                        pf.write(f"{fact_line}\n")
                        pf.write(f"原罪名: {orig_acc_line} 原法条: {orig_rel_line}\n")
                        pf.write(f"预测罪名: {pred_acc_line} 预测法条: {pred_rel_line}\n\n")
        y_true = np.vstack(y_true)
        y_pred = np.vstack(y_pred)
        micro, macro, val_acc = compute_metrics(y_true, y_pred)

        # collect metrics for plotting
        epoch_losses.append(epoch_loss)
        epoch_micro_f1.append(micro)
        epoch_macro_f1.append(macro)
        epoch_accs.append(val_acc)

        # --- plotting: Loss / F1 / Accuracy curves ---
        try:
            epochs_range = list(range(1, len(epoch_losses) + 1))
            fig, axs = plt.subplots(3, 1, figsize=(8, 12))
            axs[0].plot(epochs_range, epoch_losses, marker='o')
            axs[0].set_title('Loss')
            axs[0].set_xlabel('Epoch')
            axs[0].set_ylabel('Loss')

            axs[1].plot(epochs_range, epoch_micro_f1, marker='o', label='Micro-F1')
            axs[1].plot(epochs_range, epoch_macro_f1, marker='x', label='Macro-F1')
            axs[1].set_title('F1 Scores')
            axs[1].set_xlabel('Epoch')
            axs[1].set_ylabel('F1')
            axs[1].legend()

            axs[2].plot(epochs_range, epoch_accs, marker='o', color='green')
            axs[2].set_title('Exact-match Accuracy')
            axs[2].set_xlabel('Epoch')
            axs[2].set_ylabel('Accuracy')

            plt.tight_layout()
            fig_path = run_dir / 'train_metrics_curves.png'
            fig.savefig(fig_path)
            plt.close(fig)
        except Exception:
            pass

        # --- 标签相关性热力图（基于验证集标签相关性） ---
        try:
            if y_true.size > 0:
                # compute Pearson correlation between label columns
                # convert to float for correlation
                corr = np.corrcoef(y_true.T.astype(float))
                fig2, ax2 = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr, cmap='vlag', center=0, xticklabels=False, yticklabels=False, ax=ax2)
                ax2.set_title('Label Correlation (validation set)')
                heatmap_path = run_dir / 'label_correlation_heatmap.png'
                fig2.savefig(heatmap_path)
                plt.close(fig2)
        except Exception:
            pass

        # --- 混淆矩阵（多标签，按标签累加得到总体 2x2 矩阵） ---
        try:
            if y_true.size > 0:
                mcm = multilabel_confusion_matrix(y_true, y_pred)
                # sum over labels to get aggregate
                cm_sum = mcm.sum(axis=0)
                # cm_sum is (2,2): [[TN_sum, FP_sum],[FN_sum, TP_sum]]
                fig3, ax3 = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm_sum, annot=True, fmt='d', cmap='Blues', ax=ax3,
                            xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
                ax3.set_title('Aggregate Confusion Matrix (summed over labels)')
                cm_path = run_dir / 'confusion_matrix.png'
                fig3.savefig(cm_path)
                plt.close(fig3)
        except Exception:
            pass

        # --- 罪名混淆矩阵（8x8）和法条混淆矩阵（动态大小） ---
        try:
            if y_true.size > 0 and len(acc_indices) > 0 and len(article_indices) > 0:
                acc_confusion, article_confusion = build_label_confusion_matrices(
                    y_true, y_pred, acc_indices, article_indices, acc_classes, article_classes)

                # 罪名混淆矩阵
                if len(acc_classes) > 0:
                    acc_cm_path = run_dir / f'confusion_matrix_accusation_epoch{epoch}.png'
                    plot_confusion_matrix(acc_confusion, acc_classes,
                                          f'Accusation Confusion Matrix (Epoch {epoch})',
                                          acc_cm_path)

                # 法条混淆矩阵
                if len(article_classes) > 0:
                    art_cm_path = run_dir / f'confusion_matrix_article_epoch{epoch}.png'
                    plot_confusion_matrix(article_confusion, article_classes,
                                          f'Article Confusion Matrix (Epoch {epoch})',
                                          art_cm_path)
        except Exception as e:
            logger.warning(f'Failed to plot confusion matrices: {e}')

        logger.info(f'Epoch {epoch}: Loss={epoch_loss:.4f} Micro-F1={micro:.4f} Macro-F1={macro:.4f} Acc={val_acc:.4f}')

        # write per-epoch metric summary into the run directory
        metrics_path = run_dir / f'epoch_{epoch}_metrics.txt'
        with open(metrics_path, 'w', encoding='utf-8') as lf:
            lf.write(f'epoch={epoch}\n')
            lf.write(f'loss={epoch_loss:.6f}\n')
            lf.write(f'micro_f1={micro:.6f}\n')
            lf.write(f'macro_f1={macro:.6f}\n')
            lf.write(f'accuracy={val_acc:.6f}\n')
            lf.write(f'timestamp={start_time}\n')

        

        # checkpoint
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt = out_dir / f'model_epoch{epoch}.pt'
        save_checkpoint({'epoch': epoch, 'model_state': model.state_dict(), 'label2idx': label2idx}, str(ckpt))

        if micro > best_micro:
            best_micro = micro
            best_ckpt = out_dir / 'best_model.pt'
            save_checkpoint({'epoch': epoch, 'model_state': model.state_dict(), 'label2idx': label2idx}, str(best_ckpt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train BERT+GCN multi-label classifier')
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--valid_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--bert_model', type=str, default='bert-base-chinese')
    parser.add_argument('--label_dim', type=int, default=256)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()
    train(args)
