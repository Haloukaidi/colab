"""Synthetic integration tests: no private data or large model weights."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
import zipfile
import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
from pycocotools import mask as mu

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'

def load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class MaterializationTests(unittest.TestCase):
    def test_raw_caption_and_original_masks_survive_tar_materialization(self):
        module = load_script('materialize_sacap_scope')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); scope = root / 'scope'; archives = root / 'archives'; output = root / 'data'
            (scope / 'manifests').mkdir(parents=True); (scope / 'sacap').mkdir(); archives.mkdir()
            caption = 'An original regional description with many words that should not be shortened before entering the official text tokenizer.'
            mask = np.zeros((16,16),np.uint8); mask[3:9,4:11]=1
            rle = mu.encode(np.asfortranarray(mask)); rle['counts']=rle['counts'].decode()
            native=[];rows=[]; contents={}
            for number,split in [(1,'train'),(2,'val')]:
                iid=f'sa_{number}'; rid=str(number+5)
                native.append({'imagename':iid+'.jpg','caption':'Unrelated global description','segments_info':[{'anno_id':int(rid),'caption':caption}]})
                rows.append({'image_id':iid,'region_id':rid,'split':split,'text':caption,'source_archive':'sa_000020.tar','image_path':f'sa1b/images/{split}/{iid}.jpg','annotation_path':f'sa1b/annotations/{split}/{iid}.json','benchmark_excluded':False})
                image=io.BytesIO();Image.new('RGB',(16,16),'red').save(image,format='JPEG')
                contents[iid+'.jpg']=image.getvalue()
                contents[iid+'.json']=json.dumps({'annotations':[{'id':int(rid),'segmentation':rle}]}).encode()
            archive=archives/'sa_000020.tar'
            with tarfile.open(archive,'w') as tar:
                for name,data in contents.items():
                    info=tarfile.TarInfo('./'+name);info.size=len(data);tar.addfile(info,io.BytesIO(data))
            pq.write_table(pa.Table.from_pylist(rows),scope/'manifests/sacap_regions.parquet')
            pq.write_table(pa.Table.from_pylist(native),scope/'sacap/native_sacap.parquet')
            for i,split in enumerate(['train','val']):
                pq.write_table(pa.Table.from_pylist([native[i]]),scope/f'sacap/{split}.parquet')
            registry={'archives':[{'name':archive.name,'bytes':archive.stat().st_size,'sha256':module.sha256(archive)}]}
            (scope/'archive_registry.json').write_text(json.dumps(registry))
            result=module.materialize(scope,archives,output,verify_archive_hash=True)
            self.assertTrue(result['training_ready'])
            self.assertEqual(result['validated_images'],2)
            self.assertEqual(result['validated_regions'],2)
            got=[json.loads(line) for line in (output/'manifests/sacap_regions.jsonl').read_text().splitlines()]
            self.assertTrue(all(r['text']==caption for r in got))
            # Idempotent data copies, still revalidated.
            self.assertTrue(module.materialize(scope,archives,output)['training_ready'])
            rows[0]['region_id']='999'
            native[0]['segments_info'][0]['anno_id']=999
            pq.write_table(pa.Table.from_pylist(rows),scope/'manifests/sacap_regions.parquet')
            pq.write_table(pa.Table.from_pylist(native),scope/'sacap/native_sacap.parquet')
            with self.assertRaisesRegex(ValueError,'Missing mask ID'):
                module.materialize(scope,archives,root/'bad')
            self.assertFalse(json.loads((root/'bad/materialization_audit.json').read_text())['training_ready'])

    def test_zip_path_safety(self):
        module=load_script('install_mini500_v2')
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); archive=root/'bad.zip'
            with zipfile.ZipFile(archive,'w') as z:z.writestr('../escape.txt','bad')
            with self.assertRaisesRegex(ValueError,'Unsafe ZIP'):
                module.safe_zip_extract(archive,root/'target')
            self.assertFalse((root/'escape.txt').exists())

    def test_reject_wrong_mini_base_hash(self):
        module=load_script('install_mini500_v2')
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); wrong=root/'wrong.zip';wrong.write_bytes(b'not the original dataset')
            with self.assertRaisesRegex(ValueError,'Unexpected base ZIP'):
                module.install(wrong,root/'patch',root/'out')
            self.assertFalse((root/'out').exists())

if __name__=='__main__':unittest.main()
