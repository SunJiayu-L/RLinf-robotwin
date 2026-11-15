from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import csv, json, sys, tempfile

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from omegaconf import OmegaConf

work = Path(sys.argv[1])
root = work.parents[1]
base = root / 'policy_mc3500_timeout300_p60_pooled'
dest = root / 'policy_mc3500_drop80_filtered'
raw = root.parent / 'datasets/rollout'
keys = ['split', 'rank', 'episode']
tag = json.loads((base / 'manifest.json').read_text())['advantage_tag']
joint = pd.read_parquet(root / 'analysis/joint_idle_audit_L8cC6l/joint_features.parquet',
                        filters=[('window', '=', 50), ('success', '=', False)])
joint = joint[joint.eligible]
idle = joint[joint[['joint','cmd_joint']].max(axis=1).le(.08)
             & joint[['grip','cmd_grip']].max(axis=1).le(.12)]
repeat = pd.read_parquet(work / 'gripper_repeat_audit/repeat_candidates.parquet')
repeat = repeat[repeat.low_arm & repeat.open_anchor.eq(.45)]
fallback = (pd.concat([idle[keys+['start']].assign(reason='idle'),
                       repeat[keys+['start']].assign(reason='repeat')])
            .sort_values(['start','reason']).drop_duplicates(keys).set_index(keys).to_dict('index'))
with (work/'suffix_video100_drop80/index.csv').open() as f:
    reviewed = {('train',int(r['rank']),int(r['episode'])): r for r in csv.DictReader(f)}
assert len(reviewed) == 100
cv2.setNumThreads(1)

def image_hit(split, rank, episode):
    path = raw / f'rank_{rank:02d}' / split / 'data/chunk-000' / f'episode_{episode:06d}.parquet'
    cam = pq.read_table(path, columns=['observation.images.cam_high'], use_threads=False).column(0)
    assert len(cam) == 400
    hits = [0,0]
    for t, item in enumerate(cam):
        img = cv2.imdecode(np.frombuffer(item.as_py()['bytes'], np.uint8), cv2.IMREAD_COLOR)
        h = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        masks = [cv2.inRange(h,(35,95,45),(95,255,255)),
                 cv2.inRange(h,(0,95,45),(11,255,255)) | cv2.inRange(h,(170,95,45),(179,255,255))]
        for i, mask in enumerate(masks):
            n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
            yes = False
            if n > 1:
                idx = 1 + np.argmax(stats[1:,4])
                yy,xx = np.where(labels == idx)
                if len(yy) >= 40 and yy.mean()/img.shape[0] >= .22:
                    _, vectors = np.linalg.eigh(np.cov(np.c_[xx, yy].T))
                    dx, dy = vectors[:,-1]
                    yes = np.degrees(np.arctan2(abs(dx), abs(dy))) >= 60
            hits[i] = hits[i] + 1 if yes else 0
        if 3 in hits:
            return t-2
    return None

def cutoff(key):
    drop = image_hit(*key)
    if drop is not None:
        cut, reason = min(400,drop+81), 'image'
    elif key in fallback:
        cut, reason = int(fallback[key]['start']), fallback[key]['reason']
    else:
        cut, reason = 400, 'no_trigger'
    if key in reviewed:
        old = reviewed[key]
        expected = int(old['new_cut'])
        if old['first_candidate']:
            assert (cut,drop) == (expected,int(old['first_candidate'])), key
        else:
            assert drop is None, key
            cut, reason = expected, old['old_group']
    return dict(zip(keys,key),cut=cut,reason=reason,drop=drop,reviewed=key in reviewed)

def publish(cuts):
    counts=[]
    lookup=cuts.set_index(keys).cut
    with tempfile.TemporaryDirectory(dir=root,prefix='.drop80_') as tmp:
        stage=Path(tmp)/'publish'
        for split in ('train','val'):
            for name in [f'sft_{split}',*(f'rank_{i:02d}/{split}' for i in range(8))]:
                src=base/'datasets'/name
                out=stage/'datasets'/name
                (out/'meta').mkdir(parents=True)
                label=f'advantages_{tag}.parquet'
                for f in src.iterdir():
                    if f.name!='meta': (out/f.name).symlink_to(f,target_is_directory=f.is_dir())
                for f in (src/'meta').iterdir():
                    if f.name!=label: (out/'meta'/f.name).symlink_to(f)
                df=pd.read_parquet(src/'meta'/label)
                eligible=df.policy_eligible.copy()
                mask=pd.Series(False,index=df.index)
                if name.startswith('rank_'):
                    rank=int(name[5:7])
                    values=df.episode_index.map(lookup.xs((split,rank),level=('split','rank')))
                    assert values.notna().equals(~df.success), name
                    mask=eligible & df.frame_index.add(49).ge(values)
                assert not (mask & df.success).any()
                counts.append(dict(dataset=name,split=split,before=int(eligible.sum()),
                                   removed=int(mask.sum()),positive_removed=int(df.loc[mask,'advantage'].sum()),
                                   after=int((eligible & ~mask).sum())))
                df['policy_eligible']=eligible & ~mask
                df['advantage']=df.advantage.astype('boolean').mask(mask)
                assert df.advantage.notna().equals(df.policy_eligible)
                df.to_parquet(out/'meta'/label,index=False)
        cfg=OmegaConf.load(base/'robotwin_pick_dual_mc3500_timeout300_pooled_30k.yaml')
        cfg.runner.logger.log_path=str(dest/'runs')
        for entry in cfg.data.train_data_paths:
            entry.dataset_path=str(dest/'datasets'/Path(entry.dataset_path).relative_to(base/'datasets'))
        OmegaConf.save(cfg,stage/'policy_drop80.yaml')
        cuts.to_csv(stage/'episode_cutoffs.csv',index=False)
        report={'base':str(base),'datasets':counts,
                'totals':pd.DataFrame(counts).drop(columns='dataset').groupby('split').sum().to_dict('index'),
                'cut_counts':cuts.groupby(['split','reason']).size().unstack(fill_value=0).to_dict('index')}
        (stage/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
        assert not dest.exists()
        stage.rename(dest)
        print(json.dumps({k:report[k] for k in ('totals','cut_counts')}),flush=True)

assert not dest.exists()
items=sorted(joint[keys].drop_duplicates().itertuples(index=False,name=None))
cache=work/'drop80_scanned.csv'
if cache.exists():
    cuts=pd.read_csv(cache)
    assert sorted(cuts[keys].itertuples(index=False,name=None))==items
else:
    with ThreadPoolExecutor(max_workers=8) as pool:
        records=[]
        for i,result in enumerate(pool.map(cutoff,items),1):
            records.append(result)
            if i%100==0: print(f'Scanned {i}/{len(items)}',flush=True)
    cuts=pd.DataFrame(records)
    temp=cache.with_suffix('.tmp')
    cuts.to_csv(temp,index=False)
    temp.replace(cache)
assert int(cuts.reviewed.sum())==100
publish(cuts)
