#!/usr/bin/env python3
"""
Filter dataset files to only keep records whose `accusation` (after splitting)
contain only items from a fixed allowed 20-list.

Usage:
  python filter_top20_accusations.py /root/autodl-tmp/dir_jcl/graduation_design/dataset_initial

This will read `test.json`, `train.json`, `evalution.json` under the directory,
and write `test_strict20.json`, `train_strict20.json`, `evalution_strict20.json`.
"""
import json
import os
import re
import sys
from pathlib import Path

ALLOWED = [
    "盗窃",
    "危险驾驶",
    "故意伤害",
    "交通肇事",
    "诈骗",
    "寻衅滋事",
    "抢劫",
    "开设赌场",
]
ALLOWED_SET = set(ALLOWED)

# split on Chinese/English commas, ideographic list marker '、', semicolons, slashes, and common conjunctions
SPLIT_RE = re.compile(r'[，,、;；/\\|\t\n\r]|\band\b|\b和\b|\b与\b|\b及\b')


def split_accusation_field(acc_field):
    """Return list of accusation strings extracted from acc_field.

    acc_field may be:
      - a list of strings (each may itself contain multiple accusations separated by '、' etc.)
      - a single string
      - nested under `meta` etc. (handled by caller)
    """
    parts = []
    if acc_field is None:
        return parts
    if isinstance(acc_field, list):
        items = acc_field
    else:
        items = [acc_field]
    for it in items:
        if not isinstance(it, str):
            it = str(it)
        for p in SPLIT_RE.split(it):
            p2 = p.strip()
            if p2:
                parts.append(p2)
    return parts


def iter_records(path):
    """Yield parsed JSON records from file. Tries line-delimited first, falls back to full-file parse."""
    with open(path, 'r', encoding='utf-8') as f:
        # try NDJSON: one json object per line
        f.seek(0)
        all_lines = f.readlines()
        per_line = []
        ok = True
        for ln in all_lines:
            s = ln.strip()
            if not s:
                continue
            try:
                per_line.append(json.loads(s))
            except Exception:
                ok = False
                break
        if ok and per_line:
            for obj in per_line:
                yield obj
            return

    # fallback: whole-file parse (array or concatenated objects)
    with open(path, 'r', encoding='utf-8') as f:
        txt = f.read().strip()
    if not txt:
        return
    try:
        obj = json.loads(txt)
        if isinstance(obj, list):
            for item in obj:
                yield item
            return
        if isinstance(obj, dict):
            # try common keys
            for key in ('data', 'records', 'items'):
                if key in obj and isinstance(obj[key], list):
                    for item in obj[key]:
                        yield item
                    return
            # else treat as single record
            yield obj
            return
    except Exception:
        pass

    # try concatenated JSON objects
    dec = json.JSONDecoder()
    idx = 0
    L = len(txt)
    while idx < L:
        try:
            obj, off = dec.raw_decode(txt[idx:])
        except Exception:
            break
        yield obj
        idx += off
        while idx < L and txt[idx].isspace():
            idx += 1


def get_acc_from_record(rec):
    # look for top-level `accusation`, then `meta.accusation`, else search nested dicts
    if not isinstance(rec, dict):
        return []
    if 'accusation' in rec:
        return split_accusation_field(rec.get('accusation'))
    meta = rec.get('meta') or rec.get('Meta') or rec.get('metadata')
    if isinstance(meta, dict) and 'accusation' in meta:
        return split_accusation_field(meta.get('accusation'))
    # fallback: search nested dict values
    for v in rec.values():
        if isinstance(v, dict) and 'accusation' in v:
            return split_accusation_field(v.get('accusation'))
    return []


def filter_file(in_path, out_path):
    kept = 0
    total = 0
    # also accumulate counts per accusation in filtered set
    counts = {}
    with open(out_path, 'w', encoding='utf-8') as outf:
        for rec in iter_records(in_path):
            total += 1
            accs = get_acc_from_record(rec)
            if not accs:
                continue
            # normalize: remove duplicates while preserving order
            seen = set()
            normalized = []
            for a in accs:
                if a not in seen:
                    seen.add(a)
                    normalized.append(a)
            # check all accusations are in allowed set
            if all((a in ALLOWED_SET) for a in normalized):
                # write record with normalized accusation array (prefer top-level placement)
                out_rec = dict(rec)
                # place normalized list at top-level `accusation`
                out_rec['accusation'] = normalized
                json.dump(out_rec, outf, ensure_ascii=False)
                outf.write('\n')
                kept += 1
                for a in normalized:
                    counts[a] = counts.get(a, 0) + 1

    return total, kept, counts


def main():
    if len(sys.argv) < 2:
        print('Usage: filter_top20_accusations.py /path/to/dataset_initial_dir')
        sys.exit(1)
    base = Path(sys.argv[1])
    if not base.is_dir():
        print('path must be a directory')
        sys.exit(1)

    files = [
        ('test.json', 'test_strict20.json'),
        ('train.json', 'train_strict20.json'),
        ('evalution.json', 'evalution_strict20.json'),
    ]

    overall_counts = {}
    for inp, outp in files:
        in_path = base / inp
        out_path = base / outp
        if not in_path.exists():
            print(f'warning: {in_path} not found, skipping')
            continue
        print(f'Processing {in_path} -> {out_path} ...')
        total, kept, counts = filter_file(str(in_path), str(out_path))
        print(f'  total={total}, kept={kept}')
        for k, v in counts.items():
            overall_counts[k] = overall_counts.get(k, 0) + v

    # print overall summary for allowed 20 (include zeros)
    print('\nOverall counts in filtered outputs:')
    for a in ALLOWED:
        print(f'  {a}: {overall_counts.get(a,0)}')


if __name__ == '__main__':
    main()
