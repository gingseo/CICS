from pathlib import Path
import pytest
from src.job_manager import JobManager,save_selection
from src.common import read_json,write_json


def test_duplicate_launch_does_not_spawn_twice(tmp_path,monkeypatch):
    launches=[]
    class Process: pid=12345
    monkeypatch.setattr('src.job_manager.subprocess.Popen',lambda *a,**k:launches.append(a) or Process())
    monkeypatch.setattr('src.job_manager.process_identity',lambda pid:'identity')
    manager=JobManager(tmp_path)
    a=manager.launch(dict(kind='analysis',fixture=True));b=manager.launch(dict(kind='analysis',fixture=True))
    assert a['job_id']==b['job_id'] and len(launches)==1
    with pytest.raises(RuntimeError): manager.launch(dict(kind='training'))


def evidence(root,phase,ratio=.5,stage='C4'):
    conditions={'A':[(r,'C3','cosine') for r in [.75,.5,.25]],
        'B':[(ratio,s,'cosine') for s in ['C3','C4','C5']],
        'C':[(ratio,stage,m) for m in ['cosine','l1','l2']]}[phase]
    paths=[]
    for i,(r,s,m) in enumerate(conditions):
        p=root/f'{phase}{i}';p.mkdir(exist_ok=True)
        write_json(p/'manifest.json',dict(checkpoint_sha256='x',dataset_fingerprint='data',split='val',ratio=r,stage=s,metric=m))
        write_json(p/'state.json',dict(status='DONE'));paths.append(str(p))
    return paths


def test_manual_selection_invalidates_downstream(tmp_path):
    path=tmp_path/'selection.json'
    save_selection(path,'A',.5,'distribution',evidence(tmp_path,'A'))
    save_selection(path,'B','C4','consistency',evidence(tmp_path,'B'))
    save_selection(path,'C','cosine','scale',evidence(tmp_path,'C'))
    state=save_selection(path,'A',.25,'reconsidered',evidence(tmp_path,'A'))
    assert 'stage' not in state and 'metric' not in state and len(state['history'])==4
    with pytest.raises(ValueError): save_selection(path,'B','C3','',[])
    with pytest.raises(ValueError): save_selection(path,'B','C3','stale condition',evidence(tmp_path,'B'))
