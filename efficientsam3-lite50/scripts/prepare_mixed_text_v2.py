"""Mix existing Recap-v3 phrases with original RefCOCO-family expressions.

Local data preparation only. No new text rewriting or model-policy changes.
Use the exact tokenizer assets bundled with the pinned public data download.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import time
import pyarrow as pa
import pyarrow.parquet as pq

RECAP_SHA = '9153dbddaf9eb3e75e0ff45b5dabfd91afb489d8df9a516a380fef1e1c892c29'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def jsonl(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def prepare(args):
    started = time.time()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    assets = Path(args.tokenizer_assets)
    spec = importlib.util.spec_from_file_location('official_tokenizer', assets / 'tokenizer_ve.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tokenizer = module.SimpleTokenizer(str(assets / 'bpe_simple_vocab_16e6.txt.gz'))

    def annotate(row):
        ids = tokenizer.encode(row['text'])
        effective = [tokenizer.sot_token_id] + ids[:14] + [tokenizer.eot_token_id]
        row['content_tokens'] = len(ids)
        row['would_truncate_ctx16'] = len(ids) > 14
        row['normalized_sha256'] = hashlib.sha256(tokenizer.clean_fn(row['text']).encode()).hexdigest()
        row['input16_sha256'] = hashlib.sha256(struct.pack('<' + 'I' * len(effective), *effective)).hexdigest()
        return row

    refroot = Path(args.refcoco_root)
    refs = {split: jsonl(refroot / f'text/refcoco_{split}.jsonl.gz')
            for split in ('train', 'val', 'test')}
    frequencies = json.loads(Path(args.recap_val_frequencies).read_text())
    holdout = []
    counts = Counter()
    for split in ('test', 'val'):
        for record in refs[split]:
            holdout.append(annotate(dict(record, source=record['dataset'],
                frequency=record.get('occurrences', 1), split=split)))
    for text, frequency in frequencies.items():
        holdout.append(annotate(dict(text=text, frequency=frequency, split='val',
            source='recap_v3', text_policy='curated_phrase_unmodified')))
    holdout_keys, holdout_normalized, retained_holdout = set(), set(), []
    for row in holdout:
        if row['input16_sha256'] in holdout_keys:
            counts['duplicate_holdout_effective_input'] += 1
            continue
        holdout_keys.add(row['input16_sha256'])
        holdout_normalized.add(row['normalized_sha256'])
        retained_holdout.append(row)
    if sha(args.recap_train) != RECAP_SHA:
        raise ValueError('Expected the frozen Recap-v3 training vocabulary')
    recap = pq.read_table(args.recap_train).to_pylist()
    if len(recap) != 1045907:
        raise ValueError('Unexpected frozen Recap training vocabulary size')
    train, seen, histogram, sources = [], set(), Counter(), Counter()
    for source, records in [('refcoco_family', refs['train']), ('recap_v3', recap)]:
        for record in records:
            row = dict(record, source=record.get('dataset', source), split='train',
                       frequency=record.get('frequency', record.get('occurrences', 1)))
            row['text_policy'] = ('curated_phrase_unmodified' if source == 'recap_v3'
                                  else 'original_runtime_ctx16')
            annotate(row)
            if source == 'recap_v3':
                if not 0 < row['content_tokens'] <= 14:
                    raise ValueError('A processed Recap-v3 phrase violates the 14-BPE contract')
                histogram[row['content_tokens']] += row['frequency']
            key = row['input16_sha256']
            if key in holdout_keys or row['normalized_sha256'] in holdout_normalized:
                counts[source + '_holdout_overlap_removed'] += 1
                continue
            if key in seen:
                counts[source + '_effective_input_deduplicated'] += 1
                continue
            seen.add(key)
            train.append(row)
            sources[row['source']] += 1
    if seen.intersection(holdout_keys):
        raise AssertionError('Effective 16-token train/holdout overlap')
    schema = pa.schema([
        ('text', pa.string()), ('split', pa.string()), ('source', pa.string()),
        ('frequency', pa.int64()), ('content_tokens', pa.int32()),
        ('would_truncate_ctx16', pa.bool_()), ('text_policy', pa.string()),
        ('normalized_sha256', pa.string()), ('input16_sha256', pa.string()),
        ('dataset', pa.string()), ('image_id', pa.int64()), ('ann_id', pa.int64()),
        ('ref_id', pa.int64()), ('sent_id', pa.int64()), ('source_split', pa.string()),
    ])
    pq.write_table(pa.Table.from_pylist(train + retained_holdout, schema=schema),
                   out / 'text_mixed_full.parquet', compression='zstd')
    mini = []
    for split, records, n in [('train', train, 4500), ('val', retained_holdout, 500)]:
        for source in ('recap_v3', 'refcoco_family'):
            candidates = [r for r in records if r['split'] == split and
                          ((r['source'] == 'recap_v3') == (source == 'recap_v3'))]
            if len(candidates) < n:
                raise ValueError('Not enough records for the development mix')
            candidates.sort(key=lambda r: hashlib.sha256(
                ('mini500-v2:' + r['input16_sha256']).encode()).hexdigest())
            mini.extend(candidates[:n])
    pq.write_table(pa.Table.from_pylist(mini, schema=schema),
                   out / 'text_mixed_mini500.parquet', compression='zstd')
    report = {
        'status': 'prepared_not_accuracy_validated',
        'recap_unique_training_phrases_retokenized': len(recap),
        'recap_weighted_content_lengths': dict(histogram),
        'recap_source_sha256': sha(args.recap_train),
        'refcoco_source_counts': {k: len(v) for k, v in refs.items()},
        'train_rows': len(train), 'train_by_source': dict(sources),
        'val_rows': sum(r['split'] == 'val' for r in retained_holdout),
        'test_rows': sum(r['split'] == 'test' for r in retained_holdout),
        'full_train_over14': sum(r['would_truncate_ctx16'] for r in train),
        'exclusions': dict(counts), 'train_holdout_effective_token_overlap': 0,
        'mini_train_rows': 9000, 'mini_val_rows': 1000, 'text_rewritten': False,
        'full_sampling': 'One visit per retained unique effective input; frequency is metadata, not weight.',
        'mini_sampling': '4500+4500 train,500+500 val; development mix,not an official ratio.',
        'holdout_policy': 'Source splits retained; normalized/effective-ctx16 train collisions removed; test priority over val.',
        'official_commit': 'bd0936c788fed8d51fa799437f05abd97b401b06',
        'seconds': time.time() - started,
        'limitations': [
            'Not the unpublished LiteText paper corpus; RF100-VL not included.',
            'Recap phrase semantic quality was sampled; no model accuracy measured.',
            'Recap/COCO perceptual image overlap has not been assessed.',
            'Long RefCOCO originals use automatic official runtime truncation.',
        ],
    }
    (out / 'text_mix_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('recap-train', 'recap-val-frequencies', 'refcoco-root', 'tokenizer-assets', 'output'):
        parser.add_argument('--' + name, required=True)
    prepare(parser.parse_args())
