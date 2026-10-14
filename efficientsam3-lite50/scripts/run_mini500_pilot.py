"""Four-phase Colab pilot, exact resume, held-out evaluation, resource/audit logs."""
import os,sys,json,time,hashlib,shutil,random,gc,threading,subprocess,traceback,datetime,math
from pathlib import Path
import argparse
p=argparse.ArgumentParser(); p.add_argument('--output',required=True); p.add_argument('--workdir',default='/content/lite50_experiment'); p.add_argument('--smoke',action='store_true'); p.add_argument('--stop-after-checkpoint',action='store_true'); a=p.parse_args()
W=Path(a.workdir).expanduser().resolve(); O=Path(a.output); D=W/'mini500'; L=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(L),str(W/'official'),str(W/'official/sam3')]
from lite50.bn_once import BatchNormOncePerImage
import numpy as np
import torch
from torch.nn import functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf
from lite50.stage3 import build_student,Lite50Distiller,Lite50SupervisedModel
from lite50.distillation import build_teacher,image_features,image_distillation,text_distillation,geometry_prompt_from_find_stage
from lite50.train_policy import apply_train_policy,build_optimizer,assert_optimizer_coverage,parameter_report,trainable_modules
from lite50.train_audit import fingerprints,frozen_keys
from lite50.train_data import ImageFeatureDataset
from lite50.data_v2 import TextManifestDataset,RawSACapStage3Dataset
from lite50.geometry_data import SA1BGeometryStage3Dataset,collate_geometry
from sam3.model.utils.misc import copy_data_to_device
from pycocotools import mask as mask_utils
from PIL import Image
import psutil
import fcntl
lock=open(W/('smoke.lock' if a.smoke else 'training.lock'),'w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)

ATTEMPT=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
O.mkdir(parents=True,exist_ok=True)
for n in ['logs','metrics','samples','hardware','checkpoints','code','data_audit']: (O/n).mkdir(exist_ok=True)
def sha(path):
    with open(path,'rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,obj):
    path=Path(path); temp=path.with_suffix(path.suffix+'.tmp'); temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)); os.replace(temp,path)
def log(name,obj):
    with (O/name).open('a') as f: f.write(json.dumps(obj,ensure_ascii=False)+'\n')
def status(**kw):
    obj={'time':time.time(),'attempt':ATTEMPT,**kw}; write(O/'status.json',obj); print(json.dumps(obj),flush=True)
# Keep the historical pilot recipe unchanged for checkpoint compatibility.
# The current runner and BN implementation are recorded separately below.
CFG={'official_commit':'bd0936c788fed8d51fa799437f05abd97b401b06','student_commit':'9e144cb340949df108f97147bfa0d73ea55d6b2d',
     'seed':123,'resolution':1008,'amp':'bfloat16','clip_norm':1.,
     'recipe_version':'lite50_l4_shared_v1','full_training_launch_allowed':False,
     'recipe_note':'Adapted shared update-based cosine recipe, not literal official reproduction. Same schedule prefix for pilot and future full training. Full image/region counts are not yet known.',
     'lr':{'vision':0.001*8/512,'text':0.005*128/512,'geometry':5e-5,'stage3':1e-4},
     'weight_decay':{'vision':.01,'text':.05,'geometry':.01,'stage3':.01},
     'accumulation':{'vision':8,'text':4,'geometry':8,'stage3':1},
     'scheduler':{'kind':'linear_warmup_then_cosine','warmup_updates':50,'minimum_lr_ratio':.01,'total_updates':{'vision':20000,'text':100000,'geometry':30000,'stage3':1016060}},
     'stage3_backbone_lr_multiplier':.1,'adam_betas':[.9,.999],'adam_eps':1e-8,'no_decay':'1D parameters and biases',
     'epochs':{'vision':4,'text':4,'geometry':1,'stage3':5},'batch':{'vision':1,'text':32,'geometry':1,'stage3':1},
     'sacap_queries_per_chunk':32,'checkpoint_max_steps':50,'checkpoint_max_seconds':120,
     'query_microbatch':4,'freeze_unused_text_projection':True,
     'validation':'all held-out records; top confidence query; no GT-based selection; original resolution',
     'confidence_threshold':.5,'mask_probability_threshold':.5,'smoke':a.smoke,
     'data_sha256':{f:sha(D/'manifests'/f) for f in ['images.jsonl','geometry.jsonl','sacap_regions.jsonl','text_mixed_v2.parquet']}}
if (O/'config.json').exists(): assert json.loads((O/'config.json').read_text())==CFG,'Config changed: use new output'
else: write(O/'config.json',CFG)
write(O/'shared_training_recipe.json',{k:v for k,v in CFG.items() if k not in ['epochs','smoke','data_sha256','full_training_launch_allowed']})
shutil.copy2(__file__,O/'code/run_mini500.py')
shutil.copy2(__file__,O/'code'/('runner_'+ATTEMPT+'.py'))
shutil.copy2(L/'lite50/bn_once.py',O/'code'/('bn_once_'+ATTEMPT+'.py'))
write(O/'code'/('provenance_'+ATTEMPT+'.json'),{
    'runner_sha256':sha(__file__), 'bn_helper_sha256':sha(L/'lite50/bn_once.py'),
    'bn_policy':'one committed running-stat update per logical image after backward',
    'recipe_commit_is_historical':True,'full_training_allowed':False})
torch.manual_seed(123); random.seed(123); np.random.seed(123); torch.set_num_threads(4)
torch.backends.cudnn.benchmark=False
PHASES=list(CFG['epochs']); running={'phase':'initializing','step':0}; stop=threading.Event()
def monitor():
    proc=psutil.Process()
    while not stop.is_set():
        try:
            gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu','--format=csv,noheader,nounits'],text=True).strip()
            log('hardware/'+ATTEMPT+'.jsonl',{'time':time.time(),'attempt':ATTEMPT,**running,'gpu_fields':['gpu_percent','memory_controller_percent','used_MiB','total_MiB','power_W','temperature_C'],'gpu_values':gpu,'cpu_percent':psutil.cpu_percent(),'process_cpu_percent':proc.cpu_percent(),'rss_bytes':proc.memory_info().rss,'system_used_bytes':psutil.virtual_memory().used,'disk_free_bytes':shutil.disk_usage('/content').free})
        except Exception as e: print('monitor',repr(e),flush=True)
        stop.wait(5)
threading.Thread(target=monitor,daemon=True).start()
def amp(): return torch.autocast('cuda',dtype=torch.bfloat16)

def schedule_factor(phase,completed_updates):
    s=CFG['scheduler']; warm=s['warmup_updates']; total=s['total_updates'][phase]
    if completed_updates<warm: return (completed_updates+1)/warm
    progress=min(1.,(completed_updates-warm)/max(1,total-warm))
    return s['minimum_lr_ratio']+(1-s['minimum_lr_ratio'])*.5*(1+math.cos(math.pi*progress))

def optimizer_for(model,phase):
    groups={}
    for name,param in model.named_parameters():
        if not param.requires_grad: continue
        scale=CFG['stage3_backbone_lr_multiplier'] if phase=='stage3' and 'trunk.model.backbone.' in name else 1.
        decay=0. if param.ndim<=1 or name.endswith('.bias') else CFG['weight_decay'][phase]
        key=(scale,decay)
        if key not in groups: groups[key]={'params':[],'lr':CFG['lr'][phase]*scale,'base_lr':CFG['lr'][phase]*scale,'weight_decay':decay}
        groups[key]['params'].append(param)
    return torch.optim.AdamW(list(groups.values()),betas=tuple(CFG['adam_betas']),eps=CFG['adam_eps'])

class Chunks:
    """Every original caption is visited; limit query batch without dropping tails."""
    def __init__(self,split):
        self.ds=RawSACapStage3Dataset(D,split=split,max_segments_per_image=CFG['sacap_queries_per_chunk'])
        self.items=[(i,start) for i,key in enumerate(self.ds.image_ids) for start in range(0,len(self.ds.groups[key]),CFG['sacap_queries_per_chunk'])]
    def __len__(self): return len(self.items)
    def set_epoch(self,e): self.ds.set_epoch(e)
    def __getitem__(self,index):
        i,start=self.items[index]; key=self.ds.image_ids[i]; rows=self.ds.groups[key]
        self.ds.groups[key]=rows[start:start+CFG['sacap_queries_per_chunk']]
        try: return self.ds[i]
        finally: self.ds.groups[key]=rows
    def rows(self,index):
        i,start=self.items[index]; return self.ds.groups[self.ds.image_ids[i]][start:start+CFG['sacap_queries_per_chunk']]
def dataset(phase,split):
    if phase=='vision': return ImageFeatureDataset(D,split)
    if phase=='text': return TextManifestDataset(D/'manifests/text_mixed_v2.parquet',split)
    if phase=='geometry': return SA1BGeometryStage3Dataset(D,split,prompt_mode='mixed')
    return Chunks(split)
def batch_for(ds,indices,phase,query_slice=None):
    if query_slice is not None:
        assert phase=='stage3' and len(indices)==1
        i,start=ds.items[indices[0]]; key=ds.ds.image_ids[i]; original=ds.ds.groups[key]
        lo,hi=query_slice; ds.ds.groups[key]=original[start+lo:start+hi]
        try: samples=[ds.ds[i]]
        finally: ds.ds.groups[key]=original
    else: samples=[ds[i] for i in indices]
    if phase=='vision': return torch.stack(samples).cuda()
    if phase=='text': return samples
    return copy_data_to_device(collate_geometry(samples)['all'],'cuda')
def frozen_set(model):
    keys=frozen_keys(model)
    if model._lite50_phase=='stage3':
        keys.update(n for n,b in model.named_buffers() if 'trunk.model.backbone.' in n)
    return keys
def gradient_audit(model,step):
    names={id(v):n for n,v in model.named_parameters()}; groups={}
    for label,module in trainable_modules(model,model._lite50_phase).items():
        params=[v for v in module.parameters() if v.requires_grad]
        missing=[names[id(v)] for v in params if v.grad is None]
        nonzero=[names[id(v)] for v in params if v.grad is not None and torch.count_nonzero(v.grad).item()>0]
        groups[label]={'parameters':sum(v.numel() for v in params),'missing_gradients':missing,'nonzero_gradient_tensors':len(nonzero)}
        assert not missing,f'Missing gradients in {label}: {missing}'
        assert nonzero,f'No gradient reaches {label}'
    assert not any(v.grad is not None for v in model.parameters() if not v.requires_grad)
    log('logs/gradient_'+ATTEMPT+'.jsonl',{'phase':model._lite50_phase,'step':step,'groups':groups,'frozen_gradients_absent':True})
    return groups

def save_checkpoint(model,opt,phase,epoch,next_batch,step,complete=False):
    state={'model':model.state_dict(),'optimizer':opt.state_dict(),'phase':phase,'epoch':epoch,'next_batch':next_batch,'step':step,'complete':complete,'config':CFG,
           'scheduler':{'completed_updates':step,'next_factor':schedule_factor(phase,step),'recipe':CFG['scheduler']},'accumulation_position':0,'scaler':None,
           'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all(),'rng_python':random.getstate(),'rng_numpy':(np.random.get_state()[0],np.random.get_state()[1].tolist(),*np.random.get_state()[2:])}
    local=W/'checkpoint_writing.pt'; torch.save(state,local)
    digest=sha(local); ck=O/'checkpoints'; slot=ck/f'resume_{step%2}.pt'
    # alternate by serial, independent of step divisibility
    previous=json.loads((ck/'latest.json').read_text()) if (ck/'latest.json').exists() else None
    serial=previous.get('serial',0)+1 if previous else 1; slot=ck/f'resume_{serial%2}.pt'
    tmp=slot.with_suffix('.tmp'); shutil.copy2(local,tmp); assert sha(tmp)==digest; os.replace(tmp,slot)
    meta={'file':slot.name,'sha256':digest,'phase':phase,'epoch':epoch,'next_batch':next_batch,'step':step,'complete':complete,'serial':serial,'time':time.time()}
    if previous: write(ck/'previous.json',previous)
    write(ck/'latest.json',meta)
    # Deserialize every saved file and compare every model tensor, not just key names.
    loaded=torch.load(slot,map_location='cpu',weights_only=True,mmap=True)
    assert all(torch.equal(v.detach().cpu(),loaded['model'][n]) for n,v in model.state_dict().items())
    assert loaded['optimizer']['param_groups']==opt.state_dict()['param_groups']
    for pid,vals in opt.state_dict()['state'].items():
        for key,v in vals.items():
            got=loaded['optimizer']['state'][pid][key]
            assert torch.equal(v.detach().cpu(),got) if torch.is_tensor(v) else v==got
    log('logs/checkpoint_checks.jsonl',{**meta,'all_model_tensors_equal':True,'optimizer_groups_equal':True,'all_optimizer_state_equal':True})
    if complete: shutil.copy2(slot,ck/f'{phase}_complete.pt')
    return meta
def resume_state():
    for filename in ['latest.json','previous.json']:
        m=O/'checkpoints'/filename
        if not m.exists(): continue
        meta=json.loads(m.read_text()); path=m.parent/meta['file']
        if path.exists() and sha(path)==meta['sha256']:
            state=torch.load(path,map_location='cpu',weights_only=True,mmap=True); assert state['config']==CFG
            status(action='resume',checkpoint=meta); return state,path
    if (O/'checkpoints/latest.json').exists(): raise RuntimeError('No intact resume checkpoint; refusing to restart distillation')
    return None,None
def restore_rng(state):
    torch.set_rng_state(state['rng_torch']); torch.cuda.set_rng_state_all(state['rng_cuda']); random.setstate(state['rng_python'])
    r=state['rng_numpy']; np.random.set_state((r[0],np.array(r[1],dtype=np.uint32),*r[2:]))

@torch.no_grad()
def reload_inference_check(model,phase):
    rng=torch.get_rng_state(); cuda_rng=torch.cuda.get_rng_state_all(); py_rng=random.getstate(); np_rng=np.random.get_state()
    checkpoint=O/'checkpoints'/f'{phase}_complete.pt'
    fresh=build_student(str(W/'efficientsam3_repvit.pt'),str(checkpoint),phase,'cuda').eval()
    model.eval(); ds=SA1BGeometryStage3Dataset(D,'val',prompt_mode='point'); batch=batch_for(ds,[0],'geometry')
    def prediction(m):
        with amp():
            v=m.backbone.forward_image(batch.img_batch); t=m.backbone.forward_text(batch.find_text_batch,device='cuda')
            out=m.forward_grounding({**v,**t},batch.find_inputs[0],None,geometry_prompt_from_find_stage(batch.find_inputs[0],True))
        return {k:out[k].detach().cpu() for k in ['pred_masks','pred_logits','pred_boxes']}
    old,new=prediction(model),prediction(fresh)
    errors={k:float((old[k].float()-new[k].float()).abs().max()) for k in old}
    assert all(torch.equal(old[k],new[k]) for k in old),errors
    write(O/'logs'/f'{phase}_reload_inference.json',{'strict_load':True,'prediction_bitwise_equal':True,'max_absolute_errors':errors})
    del fresh,batch,old,new; model.train(); gc.collect(); torch.cuda.empty_cache()
    torch.set_rng_state(rng); torch.cuda.set_rng_state_all(cuda_rng); random.setstate(py_rng); np.random.set_state(np_rng)

@torch.no_grad()
def eval_features(model,teacher,phase,tag):
    ds=dataset(phase,'val'); model.eval(); sums={}; count=0
    bs=CFG['batch'][phase]
    for start in range(0,len(ds),bs):
        ids=list(range(start,min(start+bs,len(ds))))
        b=batch_for(ds,ids,phase)
        with amp():
            if phase=='vision':
                tf,tt=image_features(teacher,b); sf,st=image_features(model,b); losses=image_distillation(sf,tf,st,tt)
            else: losses=text_distillation(model.backbone.forward_text(b,device='cuda'),teacher.backbone.forward_text(b,device='cuda'))
        for k,v in losses.items(): sums[k]=sums.get(k,0)+float(v)*len(ids)
        count+=len(ids)
        if a.smoke: break
    result={'phase':phase,'tag':tag,'records':count,**{k:v/count for k,v in sums.items()}}
    log('metrics/feature_validation.jsonl',result); model.train(); return result

@torch.no_grad()
def eval_segmentation(model,tag):
    target=O/'metrics'/f'{tag}_segmentation.json'
    if target.exists(): return json.loads(target.read_text())
    model.eval(); summary={}; sampledir=O/'samples'/tag; sampledir.mkdir(exist_ok=True)
    for mode in ['text','point','box']:
        ds=Chunks('val') if mode=='text' else SA1BGeometryStage3Dataset(D,'val',prompt_mode=mode)
        records=[]; last_id=None; visual=None
        for idx in range(len(ds)):
            rows=ds.rows(idx) if mode=='text' else [ds.rows[idx]]
            image_id=str(rows[0]['image_id']); batch=batch_for(ds,[idx],'stage3' if mode=='text' else 'geometry')
            with amp():
                if image_id!=last_id: visual=model.backbone.forward_image(batch.img_batch); last_id=image_id
                lang=model.backbone.forward_text(batch.find_text_batch,device='cuda')
                inp=batch.find_inputs[0]
                prompt=model._get_dummy_prompt(len(inp.img_ids)) if mode=='text' else geometry_prompt_from_find_stage(inp,require_geometry=True)
                pred=model.forward_grounding({**visual,**lang},inp,None,prompt)
            scores=(pred['pred_logits'].sigmoid()*pred['presence_logit_dec'].sigmoid().unsqueeze(1)).squeeze(-1)
            assert len(rows)==scores.shape[0],(len(rows),scores.shape)
            for q,row in enumerate(rows):
                best=int(scores[q].argmax()); score=float(scores[q,best]); gt=mask_utils.decode(row['segmentation']).astype(bool)
                logits=pred['pred_masks'][q,best].float()
                mask=(F.interpolate(logits[None,None],size=gt.shape,mode='bilinear',align_corners=False)[0,0]>0).cpu().numpy()
                inter=int((mask&gt).sum()); union=int((mask|gt).sum()); area=int(mask.sum()); gtarea=int(gt.sum())
                iou=inter/max(union,1); dice=2*inter/max(area+gtarea,1); accepted=score>.5
                rec={'image_id':image_id,'region_id':str(row['region_id']),'mode':mode,'score':score,'query':best,'iou_top1':iou,'dice_top1':dice,'iou_thresholded':iou if accepted else 0.,'dice_thresholded':dice if accepted else 0.,'empty_top1':area==0,'empty_thresholded':not accepted or area==0,'pred_pixels':area,'gt_pixels':gtarea,'intersection':inter,'union':union,'text':row.get('text')}
                records.append(rec)
                if len(records)<=6:
                    name=f'{mode}_{image_id}_{row["region_id"]}'
                    Image.fromarray(mask.astype(np.uint8)*255).save(sampledir/(name+'_prediction.png'))
                    Image.fromarray(gt.astype(np.uint8)*255).save(sampledir/(name+'_gt.png'))
                    im=Image.open(D/row['image_path']).convert('RGB'); im.thumbnail((768,768)); arr=np.array(im).copy()
                    pm=np.array(Image.fromarray(mask).resize(im.size,Image.Resampling.NEAREST)); gm=np.array(Image.fromarray(gt).resize(im.size,Image.Resampling.NEAREST))
                    arr[pm]=(arr[pm]*.55+np.array([255,50,50])*.45).astype(np.uint8); arr[gm]=(arr[gm]*.55+np.array([50,220,80])*.45).astype(np.uint8)
                    Image.fromarray(arr).save(sampledir/(name+'_overlay_red_pred_green_gt.jpg')); write(sampledir/(name+'.json'),rec)
            del pred,batch
            if idx%20==0: status(action='validation',tag=tag,mode=mode,completed=idx+1,total=len(ds))
            if a.smoke: break
        write(O/'metrics'/f'{tag}_{mode}_records.json',records)
        summary[mode]={'records':len(records),'images':len({r['image_id'] for r in records}),**{k:float(np.mean([r[k] for r in records])) for k in ['iou_top1','dice_top1','iou_thresholded','dice_thresholded','empty_top1','empty_thresholded']},'success_iou_ge_0.5':float(np.mean([r['iou_thresholded']>=.5 for r in records]))}
        del visual; torch.cuda.empty_cache()
    write(target,summary); model.train(); return summary

def main():
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    torch.distributed.init_process_group('nccl',init_method='file://'+str(W/('dist_'+ATTEMPT)),rank=0,world_size=1,device_id=torch.device('cuda:0'))
    state,path=resume_state()
    current=state['phase'] if state else 'vision'; index=PHASES.index(current)+(1 if state and state['complete'] else 0)
    status(action='build_student',resume=bool(state))
    student=build_student(str(W/'efficientsam3_repvit.pt'),str(path) if path else None,current,'cuda')
    if not (O/'metrics/initial_segmentation.json').exists(): eval_segmentation(student,'initial')
    teacher=None
    for pi,phase in enumerate(PHASES):
        if pi<index: continue
        running.update(phase=phase,step=0); apply_train_policy(student,phase)
        # Pooled CLIP output is unused by token-feature KD; it has no loss path.
        if phase=='text': student.backbone.language_backbone.encoder.projection_layer.requires_grad_(False)
        opt=optimizer_for(student,phase); assert_optimizer_coverage(student,opt)
        ds=dataset(phase,'train'); bs=CFG['batch'][phase]; batches=(len(ds)+bs-1)//bs
        assert len(ds)<={'vision':450,'text':9000,'geometry':3647,'stage3':58}[phase], 'Pilot-only data guard: refusing a larger training set'
        epoch0=offset=step=0
        pending_rng=state
        if state and not state['complete'] and state['phase']==phase:
            opt.load_state_dict(state['optimizer']); epoch0,offset,step=state['epoch'],state['next_batch'],state['step']
            assert state['accumulation_position']==0
            assert state['scheduler']['next_factor']==schedule_factor(phase,step)
            log('logs/resume_checks.jsonl',{'phase':phase,'step':step,'next_batch':offset,'scheduler_exact':True,'optimizer_restored':True,'partial_gradients_absent':True})
        state=None
        before=fingerprints(student); frozen=frozen_set(student)
        write(O/'logs'/f'{phase}_parameters.json',{**parameter_report(student),'optimizer_exact_coverage':True,'train_records':len(ds),'batches_per_epoch':batches,'epochs':CFG['epochs'][phase]})
        if phase!='stage3' and teacher is None:
            status(action='build_teacher',phase=phase); teacher=build_teacher(str(W/'sam3.pt'),'cuda')
        if phase=='stage3' and teacher is not None: del teacher; teacher=None; gc.collect(); torch.cuda.empty_cache()
        teacher_before=fingerprints(teacher) if teacher is not None else None
        wrapper=loss_fn=None
        if phase in ['geometry','stage3']:
            cfg=OmegaConf.load(L/'configs'/f'distill_{phase}.yaml')
            loss_fn=instantiate(cfg.trainer.loss.all).cuda()
            wrapper=Lite50Distiller(student,teacher,image_weight=.25,text_weight=0.,mask_kd_weight=.5,min_teacher_iou=.3) if phase=='geometry' else Lite50SupervisedModel(student,phase)
        elif offset==0 and epoch0==0: eval_features(student,teacher,phase,'before')
        if pending_rng is not None: restore_rng(pending_rng)
        pending_rng=None
        saved_at=time.monotonic(); seen_grad=set()
        for epoch in range(epoch0,CFG['epochs'][phase]):
            if hasattr(ds,'set_epoch'): ds.set_epoch(epoch)
            order=torch.randperm(len(ds),generator=torch.Generator().manual_seed(123+epoch)).tolist()
            student.train()
            accumulation=CFG['accumulation'][phase]
            # Smoke exercises the largest Stage3 query batch, not a convenient small image.
            if a.smoke and phase=='stage3': order=sorted(order,key=lambda i:len(ds.rows(i)),reverse=True)
            for bi in range(offset if epoch==epoch0 else 0,batches,accumulation):
                end_bi=min(bi+accumulation,batches)
                group_ids=[order[j*bs:(j+1)*bs] for j in range(bi,end_bi)]
                total_records=sum(map(len,group_ids)); start=time.monotonic(); opt.zero_grad(set_to_none=True)
                for group in opt.param_groups: group['lr']=group['base_lr']*schedule_factor(phase,step)
                torch.cuda.reset_peak_memory_stats(); mean_loss=0.; components={}
                micro_work=[(ids,None,len(ids)/total_records) for ids in group_ids]
                if phase=='stage3':
                    assert len(group_ids)==1 and len(group_ids[0])==1
                    queries=len(ds.rows(group_ids[0][0])); cap=CFG['query_microbatch']
                    micro_work=[(group_ids[0],(j,min(j+cap,queries)),(min(j+cap,queries)-j)/queries) for j in range(0,queries,cap)]
                bn_once=BatchNormOncePerImage(student) if phase=="stage3" else None
                for ids,query_slice,weight in micro_work:
                    batch=batch_for(ds,ids,phase,query_slice)
                    with amp():
                        if phase=='vision':
                            with torch.no_grad(): tf,tt=image_features(teacher,batch)
                            sf,st=image_features(student,batch); losses=image_distillation(sf,tf,st,tt); loss=losses['loss']
                        elif phase=='text':
                            with torch.no_grad(): tf=teacher.backbone.forward_text(batch,device='cuda')
                            sf=student.backbone.forward_text(batch,device='cuda'); losses=text_distillation(sf,tf); loss=losses['loss']
                        else:
                            outputs=wrapper(batch); targets=[student.back_convert(t) for t in batch.find_targets]
                            losses=loss_fn(outputs,targets); loss=losses['core_loss']
                    assert torch.isfinite(loss).all(),f'Nonfinite loss {phase}/{step}'
                    # Upstream core loss scales by sqrt(find batch size). Preserve
                    # the logical image's scale when splitting its prompt queries.
                    if phase=='stage3' and loss_fn.supervised.scale_by_find_batch_size:
                        loss=loss*math.sqrt(queries/(query_slice[1]-query_slice[0]))
                    (loss*weight).backward(); mean_loss+=float(loss.detach())*weight
                    if bn_once is not None: bn_once.after_backward()
                    for k,v in losses.items():
                        if torch.is_tensor(v) and v.numel()==1: components[k]=components.get(k,0.)+float(v.detach())*weight
                    del batch,loss,losses
                    if phase in ['geometry','stage3']: del outputs,targets
                norm=torch.nn.utils.clip_grad_norm_([v for v in student.parameters() if v.requires_grad],CFG['clip_norm'],error_if_nonfinite=True)
                if step==0 or end_bi==batches or step%100==0: gradient_audit(student,step)
                opt.step(); step+=1; running.update(step=step,epoch=epoch,batch=bi)
                rec={'attempt':ATTEMPT,'phase':phase,'epoch':epoch,'batch':bi,'step':step,'records':total_records,'microbatches':len(group_ids),'lr':[g['lr'] for g in opt.param_groups],'loss':mean_loss,'gradient_norm_before_clip':float(norm),'seconds':time.monotonic()-start,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved(),'components':components}
                log('logs/train_'+ATTEMPT+'.jsonl',rec)
                if step%10==0 or step==1: status(action='training',**{k:rec[k] for k in ['phase','epoch','step','batch','loss','seconds','lr']},batches_per_epoch=batches)
                if step%50==0 or time.monotonic()-saved_at>=120 or end_bi==batches or step==1 or a.stop_after_checkpoint:
                    save_checkpoint(student,opt,phase,epoch,end_bi,step); saved_at=time.monotonic()
                if a.stop_after_checkpoint:
                    status(action='diagnostic_checkpoint_stop',phase=phase,step=step); return
                if a.smoke: break
            offset=0
            if phase in ['vision','text']: eval_features(student,teacher,phase,f'epoch_{epoch+1}')
            save_checkpoint(student,opt,phase,epoch+1,0,step)
            if a.smoke: break
        after=fingerprints(student); changed={n for n in before if before[n]!=after[n]}; assert not changed&frozen,sorted(changed&frozen)
        if teacher is not None: assert fingerprints(teacher)==teacher_before and not any(p.grad is not None for p in teacher.parameters())
        write(O/'logs'/f'{phase}_state_audit_{ATTEMPT}.json',{'frozen_tensors_checked':len(frozen),'frozen_unchanged':True,'changed_tensors':sorted(changed),'teacher_unchanged':teacher is not None,'teacher_disabled':teacher is None,'optimizer_exact_coverage':True})
        save_checkpoint(student,opt,phase,CFG['epochs'][phase],0,step,complete=True)
        reload_inference_check(student,phase)
        if phase=='geometry': eval_segmentation(student,'after_geometry')
        del wrapper,loss_fn,opt; gc.collect(); torch.cuda.empty_cache()
    results=eval_segmentation(student,'final')
    summarizer=O/'code/summarize_mini500.py'
    if summarizer.exists(): subprocess.run([sys.executable,str(summarizer),str(O)],check=True)
    write(O/'metrics/FINAL_RESULTS.json',{'status':'completed','smoke':a.smoke,'segmentation':results,'config':CFG})
    status(action='completed',results=results)
try: main()
except BaseException as exc:
    status(action='failed',error=repr(exc),traceback=traceback.format_exc()); raise
finally:
    stop.set()
    if torch.distributed.is_initialized(): torch.distributed.destroy_process_group()
