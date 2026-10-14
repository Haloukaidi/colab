"""Materialize a prepared SACap scope from existing LOCAL SA-1B TAR archives.

Does not download data, shorten text, train models, or change training policy.
Only SACap-matching images are extracted. The full source TARs remain intact.
A successful manifest is written only after every selected JPEG and mask is
validated. This may read a large amount of data: use a local/Colab data disk.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile

import numpy as np
from PIL import Image
import pyarrow.parquet as pq
from pycocotools import mask as masks


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def copy_member(tar, member, destination):
    if not member.isfile():
        raise ValueError('Selected archive member must be a regular file')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.partial')
    with tar.extractfile(member) as source, temporary.open('wb') as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
    if temporary.stat().st_size != member.size:
        raise ValueError('Incomplete TAR member copy')
    digest = sha256(temporary)
    if destination.exists():
        if sha256(destination) != digest:
            raise FileExistsError(f'Refusing to replace different existing data: {destination}')
        temporary.unlink()
    else:
        temporary.replace(destination)
    return digest


def materialize(scope_dir, archives_root, output, verify_archive_hash=False):
    scope_dir, archives_root, output = map(Path, (scope_dir, archives_root, output))
    output.mkdir(parents=True, exist_ok=True)
    caption_path = scope_dir / 'manifests/sacap_regions.parquet'
    rows = pq.read_table(caption_path).to_pylist()
    rows = [r for r in rows if not r.get('benchmark_excluded', False)]
    image_rows, groups = defaultdict(list), defaultdict(set)
    native = {}
    for image in pq.read_table(scope_dir / 'sacap/native_sacap.parquet').to_pylist():
        for region in image['segments_info']:
            key = (Path(image['imagename']).stem, str(region['anno_id']))
            if key in native:
                raise ValueError('Duplicate original SACap region')
            native[key] = region['caption']
    seen = set()
    for row in rows:
        image_id, region_id = row['image_id'], str(row['region_id'])
        if not re.fullmatch(r'sa_\d+', image_id) or row['split'] not in ('train', 'val'):
            raise ValueError('Unsafe image ID or invalid split')
        if not re.fullmatch(r'sa_\d{6}\.tar', row['source_archive']):
            raise ValueError('Unsafe source archive name')
        key = (image_id, region_id)
        if key in seen or native.get(key) != row['text']:
            raise ValueError('Duplicate or changed original regional caption')
        seen.add(key)
        if image_rows[image_id] and any(row[k] != image_rows[image_id][0][k]
                                        for k in ('split', 'source_archive')):
            raise ValueError('Image crosses splits or archives')
        image_rows[image_id].append(row)
        groups[row['source_archive']].add(image_id)
    if not rows:
        raise ValueError('No selected source regions')
    registry = {r['name']: r for r in json.loads(
        (scope_dir / 'archive_registry.json').read_text())['archives']}
    report = {'status': 'in_progress', 'training_ready': False,
              'caption_sha256': sha256(caption_path), 'completed_archives': [],
              'validated_images': 0, 'validated_regions': 0,
              'caption_policy': 'original_runtime_ctx16', 'archive_hashes': {}}
    write_json(output / 'materialization_audit.json', report)
    inventory = []
    try:
        for archive_name, wanted in sorted(groups.items()):
            paths = list(archives_root.rglob(archive_name))
            if len(paths) != 1:
                raise FileNotFoundError(f'Expected one local {archive_name}; found {len(paths)}')
            archive = paths[0]
            expected = registry[archive_name]
            if expected.get('bytes') and archive.stat().st_size != expected['bytes']:
                raise ValueError(f'Archive size mismatch: {archive_name}')
            if verify_archive_hash:
                actual = sha256(archive)
                if expected.get('sha256') and actual != expected['sha256']:
                    raise ValueError(f'Archive checksum mismatch: {archive_name}')
                report['archive_hashes'][archive_name] = actual
            copied = {}
            with tarfile.open(archive, 'r:*') as tar:
                for member in tar:
                    name = Path(member.name).name
                    stem, suffix = Path(name).stem, Path(name).suffix.lower()
                    if stem not in wanted or suffix not in ('.jpg', '.json'):
                        continue
                    key = (stem, suffix)
                    if key in copied:
                        raise ValueError(f'Duplicate selected archive member: {key}')
                    split = image_rows[stem][0]['split']
                    directory = 'images' if suffix == '.jpg' else 'annotations'
                    destination = output / f'sa1b/{directory}/{split}/{stem}{suffix}'
                    copied[key] = copy_member(tar, member, destination)
            for image_id in sorted(wanted):
                if (image_id, '.jpg') not in copied or (image_id, '.json') not in copied:
                    raise ValueError(f'Missing image or original annotation: {image_id}')
                split = image_rows[image_id][0]['split']
                image_path = output / f'sa1b/images/{split}/{image_id}.jpg'
                annotation_path = output / f'sa1b/annotations/{split}/{image_id}.json'
                with Image.open(image_path) as image:
                    image.load()
                    width, height = image.size
                original = json.loads(annotation_path.read_text())['annotations']
                annotations = {str(a['id']): a for a in original}
                if len(annotations) != len(original):
                    raise ValueError(f'Duplicate SA-1B annotation IDs: {image_id}')
                for row in image_rows[image_id]:
                    annotation = annotations.get(str(row['region_id']))
                    if annotation is None:
                        raise ValueError(f'Missing mask ID: {image_id}/{row["region_id"]}')
                    rle = annotation['segmentation']
                    decoded = masks.decode(rle)
                    if decoded.ndim != 2 or decoded.shape != (height, width):
                        raise ValueError('Mask/image dimensions mismatch')
                    area = int(decoded.sum())
                    if area <= 0 or area != int(masks.area(rle)):
                        raise ValueError('Empty or inconsistent original mask')
                    x, y, w, h = masks.toBbox(rle).tolist()
                    if not (0 <= x < x+w <= width and 0 <= y < y+h <= height):
                        raise ValueError('Mask box is outside image')
                    if row.get('segmentation') is not None and row['segmentation'] != rle:
                        raise ValueError('Prepared mask differs from original')
                    report['validated_regions'] += 1
                inventory.append({'image_id': image_id, 'split': split,
                    'width': width, 'height': height, 'source_archive': archive_name,
                    'image_path': str(image_path.relative_to(output)),
                    'annotation_path': str(annotation_path.relative_to(output)),
                    'image_sha256': copied[(image_id, '.jpg')],
                    'annotation_sha256': copied[(image_id, '.json')]})
                report['validated_images'] += 1
            report['completed_archives'].append(archive_name)
            write_json(output / 'materialization_audit.json', report)
            print(json.dumps({'archive': archive_name, 'validated_images': report['validated_images']}), flush=True)
        if report['validated_images'] != len(image_rows) or report['validated_regions'] != len(rows):
            raise ValueError('Incomplete validation')
        (output / 'manifests').mkdir(exist_ok=True)
        for filename, records in [('images.jsonl', inventory), ('sacap_regions.jsonl', rows)]:
            temporary = output / 'manifests' / (filename + '.partial')
            temporary.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
            temporary.replace(output / 'manifests' / filename)
        (output / 'sacap').mkdir(exist_ok=True)
        for split in ('train', 'val'):
            source = scope_dir / f'sacap/{split}.parquet'
            target = output / f'sacap/{split}.parquet'
            if target.exists() and sha256(target) != sha256(source):
                raise FileExistsError('Existing SACap split differs; use a fresh output directory')
            if not target.exists():
                shutil.copy2(source, target)
        report.update(status='passed', training_ready=True,
                      scope='SACap-matching Stage3 media only, not all SA-1B archive images')
    except Exception as error:
        report.update(status='failed', training_ready=False, error=repr(error))
        write_json(output / 'materialization_audit.json', report)
        raise
    write_json(output / 'materialization_audit.json', report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scope-dir', required=True)
    parser.add_argument('--archives-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--verify-archive-hash', action='store_true')
    args = parser.parse_args()
    print(json.dumps(materialize(**vars(args)), indent=2))


if __name__ == '__main__':
    main()
