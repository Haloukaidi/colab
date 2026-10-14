"""Download pinned public SACap captions; preserve every original region text.

No images, masks, SAM3 weights, or private Drive files are downloaded here.
Scope membership is metadata only until checked against the local SA-1B TARs.
The 50-shard scope is the user's fixed sample, NOT the author's unpublished 5P.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import pyarrow as pa
import pyarrow.parquet as pq

REVISION = '439a0ff25ccda363670af5118806cac1fd08a4c0'
EXPECTED_SHA256 = 'b99ace8192bb7cb72095c71f474eda2db5358b67b184dedbd07538d7ffe48cb6'
EXPECTED_BYTES = 414264542
FIVE_PERCENT_IDS = [15,31,55,70,88,101,121,151,175,181,200,231,246,279,296,304,335,340,360,387,407,424,446,471,494,519,529,540,564,588,613,631,646,669,687,711,736,743,765,794,802,835,850,870,891,909,930,954,967,987]


def file_sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--input', help='Already downloaded unmodified anno_train.parquet')
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    original = Path(args.input) if args.input else out / 'source_anno_train.parquet'
    url = f'https://huggingface.co/datasets/0xLDF/SACap-1M/resolve/{REVISION}/annotations/anno_train.parquet'
    if not original.exists():
        request = urllib.request.Request(url, headers={'User-Agent': 'Lite50-data-release/2'})
        temporary = original.with_suffix('.partial')
        with urllib.request.urlopen(request, timeout=180) as response, temporary.open('wb') as target:
            while chunk := response.read(4 * 1024 * 1024):
                target.write(chunk)
        temporary.replace(original)
    if original.stat().st_size != EXPECTED_BYTES or file_sha256(original) != EXPECTED_SHA256:
        raise ValueError('SACap source checksum mismatch; no derived data written')
    groups = {
        'official1pct': {f'sa_{i:06d}' for i in range(20, 30)},
        'selected5pct': {f'sa_{i:06d}' for i in FIVE_PERCENT_IDS},
    }
    selected = {scope: [] for scope in groups}
    image_keys = set(); source_images = source_regions = 0
    for batch in pq.ParquetFile(original).iter_batches(batch_size=4096):
        for record in batch.to_pylist():
            name = str(record['imagename'])
            if name in image_keys:
                raise ValueError('Duplicate SACap source image')
            image_keys.add(name); source_images += 1
            regional_ids = set()
            for region in record['segments_info']:
                rid = int(region['anno_id'])
                if rid in regional_ids or not isinstance(region['caption'], str) or not region['caption'].strip():
                    raise ValueError('Duplicate/empty regional source')
                regional_ids.add(rid)
            source_regions += len(regional_ids)
            for scope, group_set in groups.items():
                if record['image_group'] in group_set:
                    selected[scope].append(record)
    reports = {}
    for scope, records in selected.items():
        directory = out / scope
        directory.mkdir(exist_ok=True)
        pq.write_table(pa.Table.from_pylist(records), directory / 'native_sacap.parquet', compression='zstd')
        manifest = []
        for image in records:
            image_id = Path(image['imagename']).stem
            # A deterministic 99/1 source-image split. Not an unpublished author split.
            value = int(hashlib.sha256(('123:' + image_id).encode()).hexdigest()[:16], 16)
            split = 'val' if value % 10000 < 100 else 'train'
            for region in image['segments_info']:
                manifest.append({
                    'image_id': image_id, 'region_id': str(region['anno_id']),
                    'image_group': image['image_group'], 'source_archive': image['image_group'] + '.tar',
                    'split': split, 'text': region['caption'],
                    'image_path': f'sa1b/images/{split}/{image_id}.jpg',
                    'annotation_path': f'sa1b/annotations/{split}/{image_id}.json',
                    'text_policy': 'original_runtime_ctx16',
                    'pairing_status': 'metadata_only_requires_local_image_and_mask_validation',
                })
        pq.write_table(pa.Table.from_pylist(manifest), directory / 'caption_manifest.parquet', compression='zstd')
        report = {'images': len(records), 'regions': len(manifest), 'archives': sorted(groups[scope]),
                  'text_rewritten': False, 'mask_validation_performed': False,
                  'native_sha256': file_sha256(directory / 'native_sacap.parquet'),
                  'caption_sha256': file_sha256(directory / 'caption_manifest.parquet'),
                  'split_rule': 'sha256(123:image_id) first64bits modulo10000 below100 => val'}
        (directory / 'scope.json').write_text(json.dumps(report, indent=2))
        reports[scope] = report
    if (reports['official1pct']['images'], reports['official1pct']['regions']) != (10388,59408):
        raise ValueError('1pct source count mismatch')
    if (reports['selected5pct']['images'], reports['selected5pct']['regions']) != (51355,292987):
        raise ValueError('5pct source count mismatch')
    sources = {'source': url, 'revision': REVISION, 'sha256': EXPECTED_SHA256,
               'source_images': source_images, 'source_regions': source_regions,
               'scopes': reports, 'training_ready': False,
               'limitation': 'Caption metadata prepared; image existence and source anno_id/RLE must be checked in local TARs.'}
    (out / 'source_and_scope_audit.json').write_text(json.dumps(sources, indent=2))
    print(json.dumps(sources, indent=2))


if __name__ == '__main__':
    main()
