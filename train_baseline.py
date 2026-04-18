import os
import argparse
import logging
from pathlib import Path
import numpy as np
import json
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from data import LegalJSONLDataset, build_label_space, build_label_cooccurrence
from models import TextCNNForMultiLabel, BiLSTMForMultiLabel, BERTForMultiLabel, BertGCNForMultiLabel
from focal_loss import FocalLoss
from utils import normalize_adj, save_checkpoint, load_jsonl


def compute_metrics(y_true, y_pred):
    micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    if y_true.shape[0] == 0:
        acc = 0.0
    else:
        acc = (y_true == y_pred).all(axis=1).mean()
    return micro, macro, acc


def collate_fn(batch):
    input_ids = torch.stack([b['input_ids'] for b in batch])
    attention_mask = torch.stack([b['attention_mask'] for b in batch])
    labels_np = np.array([b['labels'] for b in batch])
    labels = torch.from_numpy(labels_np).to(torch.float32)
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


def get_model(args, num_labels, vocab_size, adj=None):
    """Create model based on args.model."""
    if args.model == 'textcnn':
        model = TextCNNForMultiLabel(
            vocab_size=vocab_size,
            embed_dim=args.embed_dim,
            num_labels=num_labels,
            filter_sizes=args.filter_sizes,
            num_filters=args.num_filters,
        )
    elif args.model == 'bilstm':
        model = BiLSTMForMultiLabel(
            vocab_size=vocab_size,
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_labels=num_labels,
        )
    elif args.model == 'bert':
        model = BERTForMultiLabel(
            bert_model=args.bert_model,
            num_labels=num_labels,
        )
    elif args.model == 'bert_gcn':
        model = BertGCNForMultiLabel(
            bert_model=args.bert_model,
            num_labels=num_labels,
            label_dim=args.label_dim,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")
    return model


def get_criterion(args):
    """Create loss function."""
    if args.loss == 'focal':
        return FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    else:
        from torch.nn import BCEWithLogitsLoss
        return BCEWithLogitsLoss()


def get_vocab_size(tokenizer_name='bert-base-chinese'):
    """Get vocab size from tokenizer."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    return len(tokenizer)


def train(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('train_baseline')

    label2idx = build_label_space(args.train_path)
    num_labels = len(label2idx)
    logger.info(f'Number of labels: {num_labels}')

    # Build adjacency matrix with variant handling
    cooc = build_label_cooccurrence(args.train_path, label2idx)

    if args.variant == 'no_norm':
        adj = cooc
    elif args.variant == 'random_adj':
        # Random shuffle of adjacency
        rng = np.random.default_rng(42)
        n = cooc.shape[0]
        temp = cooc.copy()
        rng.shuffle(temp)
        adj = temp
    else:
        adj = normalize_adj(cooc)

    adj = torch.tensor(adj, dtype=torch.float32)

    # Create datasets
    train_ds = LegalJSONLDataset(args.train_path, label2idx, max_length=args.max_length)
    valid_ds = LegalJSONLDataset(args.valid_path, label2idx, max_length=args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Get vocab size for CNN/LSTM models
    vocab_size = get_vocab_size(args.bert_model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    # Create model
    model = get_model(args, num_labels, vocab_size, adj)
    model.to(device)

    # For GCN models, adj needs to be on device
    if args.model == 'bert_gcn':
        adj = adj.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    criterion = get_criterion(args)

    best_micro = 0.0
    start_time = datetime.now().strftime('%Y%m%d%H%M%S')

    # Output directory: checkpoints/{model}_{loss}_{variant}
    run_name = f"{args.model}_{args.loss}_{args.variant}"
    logs_root = Path(args.output_dir) / 'logs'
    logs_root.mkdir(parents=True, exist_ok=True)
    run_dir = logs_root / f"{run_name}_{start_time}"
    run_dir.mkdir(parents=True, exist_ok=True)

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

            if args.model == 'bert_gcn':
                logits = model(input_ids=input_ids, attention_mask=attention_mask, adj=adj)
            else:
                logits = model(input_ids=input_ids, attention_mask=attention_mask)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * input_ids.size(0)
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs >= args.threshold).int().cpu().numpy()
                labs = labels.cpu().numpy().astype(int)
                batch_size = labs.shape[0]
                processed += batch_size
                correct += int((labs == preds).all(axis=1).sum())

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/processed:.4f}'})

        epoch_loss = running_loss / len(train_ds)

        # Validation
        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc='Valid', leave=False):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].cpu().numpy()

                if args.model == 'bert_gcn':
                    logits = model(input_ids=input_ids, attention_mask=attention_mask, adj=adj)
                else:
                    logits = model(input_ids=input_ids, attention_mask=attention_mask)

                probs = torch.sigmoid(logits).cpu().numpy()
                preds = (probs >= args.threshold).astype(int)
                y_true.append(labels)
                y_pred.append(preds)

        y_true = np.vstack(y_true)
        y_pred = np.vstack(y_pred)
        micro, macro, val_acc = compute_metrics(y_true, y_pred)

        epoch_losses.append(epoch_loss)
        epoch_micro_f1.append(micro)
        epoch_macro_f1.append(macro)
        epoch_accs.append(val_acc)

        # Plot curves
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
            fig.savefig(run_dir / 'train_metrics_curves.png')
            plt.close(fig)
        except Exception as e:
            logger.warning(f'Failed to plot: {e}')

        logger.info(f'Epoch {epoch}: Loss={epoch_loss:.4f} Micro-F1={micro:.4f} Macro-F1={macro:.4f} Acc={val_acc:.4f}')

        # Save metrics
        metrics_path = run_dir / f'epoch_{epoch}_metrics.txt'
        with open(metrics_path, 'w', encoding='utf-8') as f:
            f.write(f'epoch={epoch}\n')
            f.write(f'loss={epoch_loss:.6f}\n')
            f.write(f'micro_f1={micro:.6f}\n')
            f.write(f'macro_f1={macro:.6f}\n')
            f.write(f'accuracy={val_acc:.6f}\n')
            f.write(f'timestamp={start_time}\n')
            f.write(f'model={args.model}\n')
            f.write(f'loss={args.loss}\n')
            f.write(f'variant={args.variant}\n')

        # Checkpoint
        out_dir = Path(args.output_dir) / run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt = out_dir / f'model_epoch{epoch}.pt'
        save_checkpoint({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'label2idx': label2idx,
            'args': vars(args),
        }, str(ckpt))

        if micro > best_micro:
            best_micro = micro
            save_checkpoint({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'label2idx': label2idx,
                'args': vars(args),
            }, str(out_dir / 'best_model.pt'))

    # Save final summary
    summary = {
        'model': args.model,
        'loss': args.loss,
        'variant': args.variant,
        'epochs': args.epochs,
        'best_micro_f1': best_micro,
        'final_micro_f1': micro,
        'final_macro_f1': macro,
        'final_accuracy': val_acc,
    }
    with open(run_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info(f'Training complete. Best Micro-F1: {best_micro:.4f}')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train multi-label classifier baselines')
    parser.add_argument('--model', type=str, required=True,
                        choices=['textcnn', 'bilstm', 'bert', 'bert_gcn'],
                        help='Model type')
    parser.add_argument('--loss', type=str, default='bce',
                        choices=['bce', 'focal'],
                        help='Loss function')
    parser.add_argument('--variant', type=str, default='full',
                        choices=['full', 'no_gcn', 'random_adj', 'no_norm'],
                        help='Ablation variant')
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--valid_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--bert_model', type=str, default='bert-base-chinese')
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--filter_sizes', type=int, nargs='+', default=[3, 4, 5])
    parser.add_argument('--num_filters', type=int, default=256)
    parser.add_argument('--label_dim', type=int, default=256)
    parser.add_argument('--focal_alpha', type=float, default=1.0)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    args = parser.parse_args()
    train(args)
