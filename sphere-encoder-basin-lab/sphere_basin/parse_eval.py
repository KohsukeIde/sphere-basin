from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd


def _parse_table_block(block: str) -> list[dict]:
    lines = [ln.rstrip() for ln in block.splitlines() if ln.strip()]
    if len(lines) < 4:
        return []
    pipe_lines = [ln for ln in lines if ln.strip().startswith('|')]
    if len(pipe_lines) < 2:
        return []
    header = [x.strip() for x in pipe_lines[0].strip('|').split('|')]
    data_rows = []
    for ln in pipe_lines[2:]:
        vals = [x.strip() for x in ln.strip('|').split('|')]
        if len(vals) != len(header):
            continue
        data_rows.append(dict(zip(header, vals)))
    return data_rows


def parse_eval_file(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    text = path.read_text(encoding='utf-8')
    blocks = re.split(r'\n(?=ckpt: )', text)
    rows: list[dict] = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        m = re.search(r'^ckpt:\s*([^,]+),\s*time:\s*(.+)$', block, flags=re.M)
        ckpt_epoch = None
        timestamp = None
        if m:
            ckpt_epoch = m.group(1).strip()
            timestamp = m.group(2).strip()
        for row in _parse_table_block(block):
            row['ckpt_epoch'] = ckpt_epoch
            row['timestamp'] = timestamp
            row['source_file'] = str(path)
            rows.append(row)
    return pd.DataFrame(rows)


def parse_eval_dir(eval_dir: str | Path) -> pd.DataFrame:
    eval_dir = Path(eval_dir)
    dfs = []
    for path in sorted(eval_dir.glob('eval_tabl_*.txt')):
        df = parse_eval_file(path)
        if not df.empty:
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)
