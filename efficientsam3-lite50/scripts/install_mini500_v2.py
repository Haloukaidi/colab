"""Install the v2 text-data overlay onto the existing, SHA-verified mini500 ZIP.

Original images, SA-1B JSON and SACap captions are not edited. The new manifest
is explicitly passed using --text-manifest during text feature distillation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import stat
import zipfile

BASE_SHA256 = '02b8dbd678cf147ab202a87b471189a30c44308020d56fd136fe368275c52f54'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def safe_zip_extract(archive, directory):
    root = Path(directory).resolve()
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            target = (root / info.filename).resolve()
            if not target.is_relative_to(root) or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError('Unsafe ZIP path or symlink')
        z.extractall(root)


def install(base_zip, patch_dir, output):
    import pyarrow.parquet as pq
    import numpy as np
    from PIL import Image
    from pycocotools import mask as masks

    base_zip, patch_dir, output = map(Path, (base_zip, patch_dir, output))
    if sha(base_zip) != BASE_SHA256:
        raise ValueError('Unexpected base ZIP; stop before extraction')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a fresh empty output directory; no existing data will be replaced')
    staging = output.parent / (output.name + '.staging')
    if staging.exists():
        raise FileExistsError('A previous staging directory exists; inspect it before retrying')
    staging.mkdir(parents=True)
    safe_zip_extract(base_zip, staging)
    candidates = list(staging.rglob('manifests/images.jsonl'))
    if len(candidates) != 1:
        raise ValueError('Base ZIP must contain one dataset manifest')
    root = candidates[0].parent.parent
    images = records(root / 'manifests/images.jsonl')
    geometry = records(root / 'manifests/geometry.jsonl')
    sacap = records(root / 'manifests/sacap_regions.jsonl')
    if (len(images), len(geometry), len(sacap)) != (500, 4049, 409):
        raise ValueError('Unexpected base sample counts')
    for name in ('images', 'geometry', 'sacap_regions'):
        if sha(root / f'manifests/{name}.jsonl') != sha(patch_dir / f'manifests/{name}.jsonl'):
            raise ValueError('Patch attempts to alter original image/region manifest')
    annotations, image_index = {}, {}
    for row in images:
        image_id = row['image_id']
        image_index[image_id] = row
        image_path = (root / row['image_path']).resolve()
        ann_path = (root / row['annotation_path']).resolve()
        if not image_path.is_relative_to(root.resolve()) or not ann_path.is_relative_to(root.resolve()):
            raise ValueError('Unsafe base manifest path')
        if sha(image_path) != row['image_sha256'] or sha(ann_path) != row['annotation_sha256']:
            raise ValueError('Source media checksum mismatch')
        with Image.open(image_path) as image:
            image.load()
            if image.size != (row['width'], row['height']):
                raise ValueError('Image dimensions mismatch')
        source = json.loads(ann_path.read_text())['annotations']
        index = {str(a['id']): a for a in source}
        if len(index) != len(source):
            raise ValueError('Duplicate original region IDs')
        annotations[image_id] = index
    for row in geometry:
        image = image_index[row['image_id']]
        if image['split'] != row['split']:
            raise ValueError('Geometry split mismatch')
        original = annotations[row['image_id']][str(row['region_id'])]['segmentation']
        if original != row['segmentation']:
            raise ValueError('Geometry RLE differs from original')
        decoded = masks.decode(original)
        if decoded.shape != (image['height'], image['width']) or int(decoded.sum()) != row['mask_area']:
            raise ValueError('Geometry shape/area mismatch')
        x, y = row['point_xy_px']
        if not (0 <= x < image['width'] and 0 <= y < image['height'] and decoded[int(y), int(x)]):
            raise ValueError('Point is not inside its target mask')
        if not np.allclose(masks.toBbox(original), row['bbox_xywh_px']):
            raise ValueError('Geometry bbox mismatch')
    native = {}
    for split in ('train', 'val'):
        for image in pq.read_table(root / f'sacap/{split}.parquet').to_pylist():
            image_id = Path(image['imagename']).stem
            if image_index[image_id]['split'] != split:
                raise ValueError('SACap split mismatch')
            for region in image['segments_info']:
                key = (image_id, str(region['anno_id']))
                if key in native:
                    raise ValueError('Duplicate native caption ID')
                native[key] = region['caption']
    for row in sacap:
        key = (row['image_id'], str(row['region_id']))
        if native.get(key) != row['text']:
            raise ValueError('SACap original regional text changed')
        if annotations[key[0]][key[1]]['segmentation'] != row['segmentation']:
            raise ValueError('SACap original mask changed')
    source_text = patch_dir / 'manifests/text_mixed_v2.parquet'
    text = pq.read_table(source_text).to_pylist()
    if sum(r['split'] == 'train' for r in text) != 9000 or sum(r['split'] == 'val' for r in text) != 1000:
        raise ValueError('Unexpected new text split')
    train = {r['input16_sha256'] for r in text if r['split'] == 'train'}
    val = {r['input16_sha256'] for r in text if r['split'] == 'val'}
    if train & val:
        raise ValueError('Text train/val effective-input overlap')
    shutil.copy2(source_text, root / 'manifests/text_mixed_v2.parquet')
    report = {'status': 'passed', 'images_checked': 500, 'geometry_regions_checked': 4049,
              'sacap_original_regions_checked': 409, 'text_train': 9000, 'text_val': 1000,
              'base_zip_sha256': BASE_SHA256, 'text_manifest_sha256': sha(source_text),
              'text_distillation_data': 'Recap-v3 + original RefCOCO-family text',
              'stage3_caption_policy': 'original_runtime_ctx16', 'model_training_performed': False}
    (root / 'DATA_V2_READY.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    if output.exists():
        output.rmdir()
    root.rename(output)
    if staging.exists() and staging != output:
        # Only empty staging parents; do not remove unexpected residual files.
        try:
            staging.rmdir()
        except OSError:
            pass
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-zip', required=True)
    parser.add_argument('--patch-dir', required=True)
    parser.add_argument('--output', required=True)
    print(json.dumps(install(**vars(parser.parse_args())), ensure_ascii=False, indent=2))
