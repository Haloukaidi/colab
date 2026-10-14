"""Data-v2 regressions without private data or pretrained model weights."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from pycocotools import mask as mu
from lite50.data_v2 import TextManifestDataset, RawSACapStage3Dataset, safe_child


class DataV2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        for d in ['sacap','manifests','sa1b/images/train','sa1b/annotations/train']:(self.root/d).mkdir(parents=True)
        self.caption='A red square on the left in a bright scene, with a detailed long description that is intentionally not shortened.'
        mask=np.zeros((16,16),np.uint8);mask[3:10,2:8]=1
        rle=mu.encode(np.asfortranarray(mask));rle['counts']=rle['counts'].decode()
        self.rle=rle
        Image.new('RGB',(16,16),'red').save(self.root/'sa1b/images/train/sa_1.jpg')
        (self.root/'sa1b/annotations/train/sa_1.json').write_text(json.dumps({'annotations':[{'id':7,'segmentation':rle}]}))
        self.rows=[{'image_id':'sa_1','region_id':'7','split':'train','text':self.caption,'segmentation':rle}]
        self.native=[{'imagename':'sa_1.jpg','caption':'This is global caption, NOT a regional target.','segments_info':[{'anno_id':7,'caption':self.caption}]}]
        self.save()

    def tearDown(self):self.tmp.cleanup()
    def save(self):
        (self.root/'manifests/sacap_regions.jsonl').write_text('\n'.join(json.dumps(r) for r in self.rows)+'\n')
        pq.write_table(pa.Table.from_pylist(self.native),self.root/'sacap/train.parquet')

    def test_text_dataset_preserves_long_strings_and_splits(self):
        p=self.root/'texts.parquet'
        pq.write_table(pa.Table.from_pylist([{'text':self.caption,'source':'refcocog','split':'train'},
                                            {'text':'short phrase','source':'recap_v3','split':'val'}]),p)
        ds=TextManifestDataset(p);self.assertEqual(len(ds),1);self.assertEqual(ds[0],self.caption)
        self.assertEqual(TextManifestDataset(p,split='val')[0],'short phrase')
        self.assertEqual(ds.data_provenance['rows'],1)

    def test_text_blank_rejected(self):
        p=self.root/'blank.parquet';pq.write_table(pa.Table.from_pylist([{'text':'  ','source':'recap_v3','split':'train'}]),p)
        with self.assertRaisesRegex(ValueError,'Blank'):TextManifestDataset(p)

    def test_raw_sacap_does_not_require_short_text_review(self):
        ds=RawSACapStage3Dataset(self.root)
        self.assertEqual(len(ds),1);self.assertEqual(ds.groups['sa_1'][0]['text'],self.caption)
        self.assertEqual(ds.data_provenance['text_policy'],'original_runtime_ctx16')

    def test_short_or_global_caption_cannot_replace_original(self):
        self.rows[0]['text']='red square';self.save()
        with self.assertRaisesRegex(ValueError,'regional text mismatch'):RawSACapStage3Dataset(self.root)
        self.rows[0]['text']=self.native[0]['caption'];self.save()
        with self.assertRaisesRegex(ValueError,'regional text mismatch'):RawSACapStage3Dataset(self.root)

    def test_duplicate_and_nontext_modes_rejected(self):
        self.rows*=2;self.save()
        with self.assertRaisesRegex(ValueError,'Duplicate'):RawSACapStage3Dataset(self.root)
        with self.assertRaisesRegex(ValueError,'text-only'):RawSACapStage3Dataset(self.root,prompt_mode='box')

    def test_paths_cannot_escape(self):
        with self.assertRaises(ValueError):safe_child(self.root,'../../other.jpg')

    def test_checkpoint_resume_rejects_changed_text_manifest(self):
        from lite50.training_utils import save_feature_checkpoint,load_feature_resume
        model=torch.nn.Linear(2,2);opt=torch.optim.AdamW(model.parameters());p=self.root/'c.pt'
        save_feature_checkpoint(p,model,opt,'text',1,123,'teacher.pt',{'manifest_sha256':'a'})
        with self.assertRaisesRegex(ValueError,'manifest changed'):
            load_feature_resume(p,model,opt,'text','teacher.pt',{'manifest_sha256':'b'})
        self.assertEqual(load_feature_resume(p,model,opt,'text','teacher.pt',{'manifest_sha256':'a'}),(1,123))

    def test_default_stage3_changes_data_not_objective(self):
        import yaml
        cfg=yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/distill_stage3.yaml').read_text())
        self.assertEqual(cfg['phase'],'stage3')
        self.assertEqual(cfg['trainer']['model']['_target_'],'lite50.stage3.build_stage3_supervised')
        self.assertNotIn('teacher_checkpoint',cfg['trainer']['model'])
        self.assertEqual(cfg['trainer']['data']['train']['dataset']['_target_'],'lite50.data_v2.RawSACapStage3Dataset')
        self.assertEqual(cfg['max_segments_per_image'],32)
        self.assertEqual(cfg['prompt_mode'],'text')
        self.assertTrue(cfg['trainer']['data']['train']['collate_fn']['with_seg_masks'])


if __name__=='__main__':unittest.main()
