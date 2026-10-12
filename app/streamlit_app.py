from pathlib import Path
import sys
import os
import json
import pandas as pd
import yaml
import streamlit as st

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
os.chdir(PROJECT)
from src.common import read_json,sha256,digest,write_json
from src.job_manager import JobManager,save_selection,gpu_lease
from src.analysis_runner import combinations,analyze_image
from src.data_audit import load_records
from src.qualitative import render,save_views
from src.image_viewer import tiny_dominant_records,draw_gt,summary_rows,color_scale,SIZE_COLORS
from PIL import Image

st.set_page_config(page_title='CICS · Feature discrepancy',layout='wide')
st.markdown('''<style>
.block-container {max-width:1500px;padding-top:2rem;}
[data-testid="stMetricValue"] {font-size:1.65rem;}
[data-testid="stSidebar"] {background-color:#f4f6f9;}
</style>''',unsafe_allow_html=True)
ROOT=Path('runs/reproduction_r50_ms20')
DEFAULT_CHECKPOINT=str((ROOT/'test_candidates/epoch_17.pth').resolve())
manager=JobManager()

@st.cache_data(show_spinner=False)
def file_hash(path,size,mtime):
    return sha256(path)

def fingerprint(path):
    p=Path(path)
    if not p.is_file():return None
    s=p.stat();return file_hash(str(p.resolve()),s.st_size,s.st_mtime_ns)

def same_path(a,b):
    return bool(a and b) and Path(a).resolve()==Path(b).resolve()

@st.cache_data(show_spinner=False)
def records_for(root,split,fmt,folder=None):
    return load_records(root,split,fmt,image_dir=folder)

with st.sidebar:
    st.subheader('분석 설정')
    analysis_config=st.text_input('Analysis config','configs/analysis_epoch17.yaml',key='analysis_profile_v2')
    config=yaml.safe_load(Path(analysis_config).read_text())
    split=st.selectbox('분석 데이터',['test','val'],key='analysis_split_v2')
    st.caption('test: 1,610장 · val: 548장')
    pre=config['preprocess']
    st.info(f"긴 변 ≤ {pre['long_edge']} / 짧은 변 {pre.get('short_edge','제한 없음')}\n\n종횡비 유지 · 32 배수 padding")
    with st.expander('모델 및 저장 경로'):
        detector=st.text_input('Detector config',str(ROOT/'eval_test/config.py'))
        checkpoint=st.text_input('Checkpoint (.pth)',DEFAULT_CHECKPOINT)
        validation=st.text_input('Validation report',str(ROOT/'eval_test/report.json'))
        results_root=st.text_input('결과 폴더 (viewer)',config['runs_dir'])
        selection_path=st.text_input('Selection file','runs/analysis_epoch17_highres/selection.json')
        device=st.selectbox('Device',['cuda:0','cpu'])
    st.caption('BG는 알려진 GT 밖의 영역입니다. YOLO 변환본에 원본 ignore 영역은 없습니다.')
    st.caption('객체 크기 구간은 원본 GT 기준이며 COCO AP의 S/M/L 구간과 다릅니다.')
    st.markdown('VT: 2–8 · T: 8–16 · S: 16–32\n\nM: 32–64 · L: 64 이상 (px)')

selection=read_json(selection_path,{'history':[]})
validation_result=read_json(validation,{})
checkpoint_hash=fingerprint(checkpoint)
valid_checkpoint=bool(checkpoint_hash and validation_result.get('status')=='DONE'
    and validation_result.get('checkpoint_sha256')==checkpoint_hash
    and validation_result.get('config_sha256')==fingerprint(detector))

def relevant_jobs():
    jobs=[]
    for j in manager.list():
        spec=read_json(Path(j['folder'])/'spec.json',{})
        if spec.get('kind')=='analysis' and same_path(spec.get('checkpoint'),checkpoint) and spec.get('split')==split and same_path(spec.get('analysis_config'),analysis_config):
            jobs.append((j,spec))
    return sorted(jobs,key=lambda pair:pair[0].get('created_at',''),reverse=True)

st.title('CICS · Feature discrepancy 분석')
st.caption('같은 고정 모델에서 원본과 해상도 열화 입력의 표현 변화를 비교합니다.')
header=st.columns([4,1])
header[0].write(f"**{Path(checkpoint).stem} · Faster R-CNN R50-FPN · {split}**")
if header[1].button('결과 새로고침',width='stretch'):st.rerun()
if not valid_checkpoint:st.error('Checkpoint와 검증 보고서/config가 일치하지 않습니다. 모델 및 저장 경로를 확인하세요.')

viewer,control,dataset,training_tab=st.tabs(['이미지 비교','실험 진행','조건별 통계','학습 기록'])
with control:
    @st.fragment(run_every='3s')
    def live_progress():
        jobs=relevant_jobs()
        signature=tuple((j['job_id'],j['status']) for j,_ in jobs)
        before=st.session_state.get('analysis_job_signature')
        st.session_state.analysis_job_signature=signature
        if before is not None and before!=signature:st.rerun()
        for col,(phase,title,key) in zip(st.columns(3),[('A','열화 강도','ratio'),('B','Feature 위치','stage'),('C','차이 측정 방식','metric')]):
            matches=[j for j,s in jobs if s.get('phase')==phase]
            status='선택 완료' if key in selection else '대기'
            if key not in selection and matches:
                status={'DONE':'결과 확인 · 선택 대기','RUNNING':'분석 중','PENDING':'준비 중','FAILED':'실패','STOPPED':'중단'}.get(matches[0]['status'],matches[0]['status'])
            with col.container(border=True):
                st.markdown(f'**{phase} · {title}**')
                st.write(status)
                st.caption(f"선택: {selection.get(key,'—')}")
        active=next(((j,s) for j,s in jobs if j['status'] in ('PENDING','RUNNING')),None)
        if active:
            job,spec=active;p=job.get('progress',{})
            st.write(f"**{spec['phase']} 단계 · {split} 전체 분석**")
            st.progress(min(1,p.get('done',0)/max(1,p.get('total',1))))
            st.caption(f"{p.get('done',0):,} / {p.get('total',0):,}장 · 경과 {p.get('elapsed_seconds',0)/60:.1f}분 · 예상 잔여 {p.get('eta_seconds',0)/60:.1f}분")
            if not p:st.caption('모델 로딩 및 데이터 식별 정보를 준비하고 있습니다.')
            if st.button('현재 분석 중단',key='active_stop'):
                manager.stop(job['job_id']);st.rerun()
        elif jobs:
            st.info('완료된 결과를 확인하고 다음 조건을 선택하세요. 다음 단계는 자동으로 선택하지 않습니다.')
        else:st.info('A단계에서 세 가지 축소 비율을 비교합니다.')
        with st.expander('작업 및 실행 로그'):
            for job,spec in jobs:
                st.write(f"{spec['phase']} · {job['status']} · {job['job_id']}")
                if job.get('error'):st.error(job['error'])
                log=Path(job['folder'])/'run.log'
                if log.exists():
                    with log.open('rb') as f:
                        f.seek(max(0,log.stat().st_size-4000));st.code(f.read().decode(errors='replace'))
    live_progress()
    st.divider()
    left,right=st.columns([1,1])
    with left:
        st.subheader('전체 데이터 통계 계산')
        st.caption('재학습하지 않습니다. 이미지 비교 탭의 VT/T 필터와 별개로, 전체 split 또는 파일명 순서의 앞 N장을 분석합니다.')
        phase=st.selectbox('단계',['A','B','C','final'])
        limit=st.number_input('전체 통계에 사용할 앞 N장 (0 = 전체)',min_value=0,value=0,step=1)
        try:
            conditions=combinations(phase,selection)
            st.dataframe(pd.DataFrame(conditions,columns=['축소 비율','Stage','Metric']),hide_index=True,width='stretch')
            ready=True
        except ValueError:
            st.info('앞 단계의 조건을 선택하면 실행할 수 있습니다.');ready=False
        if st.button('전체 통계 계산 시작 / 재개',type='primary',disabled=not(valid_checkpoint and ready)):
            try:
                if Path(results_root).resolve()!=Path(config['runs_dir']).resolve():
                    raise ValueError('결과 폴더를 바꾸려면 Analysis config의 runs_dir도 같은 경로로 설정하세요.')
                spec=dict(kind='analysis',analysis_config=analysis_config,detector_config=detector,checkpoint=checkpoint,
                          validation=validation,phase=phase,selection=selection,split=split,limit=int(limit) or None,device=device)
                manager.launch(spec,resume=True);st.rerun()
            except Exception as e:st.error(str(e))
    with right:
        st.subheader('결과 확인 후 다음 조건 선택')
        completed=[(j,s) for j,s in relevant_jobs() if j['status']=='DONE' and s.get('phase')!='final' and not s.get('limit')]
        if not completed:st.caption('전체 split 분석이 끝나면 다음 단계 조건을 선택할 수 있습니다.')
        else:
            index=st.selectbox('선택 근거 분석 job',range(len(completed)),format_func=lambda i:f"{completed[i][1]['phase']} · {completed[i][0]['job_id']}")
            evidence,spec=completed[index];p=spec['phase']
            choices={'A':[.75,.5,.25],'B':['C3','C4','C5'],'C':['cosine','l1','l2']}[p]
            value=st.selectbox(f'{p} 결과에서 선택',choices)
            reason='GUI에서 조건 선택'
            if st.button('선택 저장 · 다음 단계 허용'):
                try:save_selection(selection_path,p,value,reason,evidence['runs']);st.rerun()
                except Exception as e:st.error(str(e))
        with st.expander('선택 이력'):st.json(selection.get('history',[]))

all_paths=[]
for p in Path(results_root).glob('*/manifest.json'):
    m=read_json(p,{})
    if m.get('checkpoint_sha256')==checkpoint_hash and m.get('split')==split:all_paths.append(p.parent)
run_paths=[p for p in all_paths if read_json(p/'state.json',{}).get('status')=='DONE']
def condition_label(path):
    m=read_json(path/'manifest.json',{})
    return f"r={m.get('ratio')} · {m.get('stage')} · {m.get('metric')} · {path.name[:6]}"

with dataset:
    st.subheader('조건별 크기 분포 비교')
    st.caption('BG는 이미지별 평균, 객체 구간은 객체별 평균입니다. Global BG 평균은 별도의 픽셀 가중 평균입니다.')
    if not run_paths:st.info('전체 분석이 끝나면 세 조건의 비교 표와 그래프가 표시됩니다. 진행 중인 이미지 결과는 이미지 비교 탭에서 볼 수 있습니다.')
    else:
        compare=st.multiselect('비교할 조건',run_paths,default=run_paths,format_func=condition_label)
        frames=[]
        for path in compare:
            frame=pd.read_csv(path/'summary.csv');frame['조건']=condition_label(path);frames.append(frame)
        if frames:
            table=pd.concat(frames,ignore_index=True)
            pivot=table.pivot(index='region',columns='조건',values='mean').reindex(['BG','underflow','VT','T','S','M','L'])
            st.dataframe(pivot.style.format('{:.5f}',na_rep='—'),width='stretch')
            st.bar_chart(pivot)
            if len({read_json(p/'manifest.json')['metric'] for p in compare})>1:st.info('Metric마다 단위가 다릅니다. raw 수치의 크기만으로 우열을 판단하지 않습니다.')
        chosen=st.selectbox('상세 결과',run_paths,format_func=condition_label)
        stats=read_json(chosen/'statistics.json')
        g=stats.get('global_bg',{})
        st.write('Global BG 평균:',g.get('mean'),'· 유효 위치 수:',g.get('n_valid_locations'))
        st.dataframe(pd.read_csv(chosen/'summary.csv'),hide_index=True,width='stretch')
        a,b=st.columns(2)
        a.image(str(chosen/'figures/bin_means.png'),caption='평균 ± 표준편차 (신뢰구간 아님)')
        b.image(str(chosen/'figures/boxplot.png'),caption='중앙값·사분위수·1.5 IQR')
        a.image(str(chosen/'figures/scatter_original_size.png'),caption='원본 객체 크기와 discrepancy')
        b.image(str(chosen/'figures/scatter_input_size.png'),caption='입력 객체 크기와 discrepancy')
        with st.expander('상관계수 및 보조 통계'):st.json(stats)
        with st.expander('CSV / Figure 다운로드'):
            for name in ['objects.csv','background.csv','summary.csv','statistics.json']:
                st.download_button(name,(chosen/name).read_bytes(),file_name=name,key='download_'+name)
            for ext in ['pdf','svg']:
                st.download_button('Boxplot '+ext,(chosen/'figures'/f'boxplot.{ext}').read_bytes(),file_name=f'boxplot.{ext}')

with viewer:
    st.subheader('VT · T 객체가 많은 이미지')
    candidates=tiny_dominant_records(records_for(config['dataset_root'],split,config['format']))
    st.caption(f"GT 객체 중 VT+T가 50% 초과인 {len(candidates):,}장 · VT+T 객체 수가 많은 순서 · 원본 파일은 변경하지 않습니다.")
    if not candidates:
        st.info('이 조건을 만족하는 이미지가 없습니다.')
    else:
        context=(config['dataset_root'],split)
        if st.session_state.get('tiny_viewer_context')!=context:
            st.session_state.tiny_viewer_context=context;st.session_state.tiny_viewer_index=0
        def move_image(step):
            st.session_state.tiny_viewer_index=(st.session_state.tiny_viewer_index+step)%len(candidates)
        left,center,right=st.columns([1,6,1])
        left.button('← 이전',on_click=move_image,args=(-1,),width='stretch')
        right.button('다음 →',on_click=move_image,args=(1,),width='stretch')
        index=st.session_state.tiny_viewer_index%len(candidates)
        record,counts=candidates[index]
        center.markdown(f"**{index+1} / {len(candidates)} · {record['image_id']}**")
        center.caption(f"VT {counts['VT']}개 + T {counts['T']}개 / 전체 {counts['total']}개 ({counts['fraction']:.1%})")
        a,b,c=st.columns(3)
        ratio=a.selectbox('축소 비율',[.75,.5,.25],key='tiny_ratio')
        stage=b.selectbox('Feature stage',['C3','C4','C5'],key='tiny_stage')
        metric=c.selectbox('Metric',['cosine','l1','l2'],key='tiny_metric')
        vmax=float(config['heatmap_max'][metric])
        st.caption('Heatmap 색은 원본과 열화 입력의 feature 차이입니다. 이미지마다 자동 정규화하지 않고 같은 metric의 색상 범위를 고정합니다.')
        identity=dict(checkpoint=checkpoint_hash,detector=fingerprint(detector),image=fingerprint(record['path']),
                      annotation=fingerprint(record['annotation_path']),preprocess=config['preprocess'],
                      degradation=config.get('degradation'),epsilon=config['epsilon'],ratio=ratio,stage=stage,metric=metric,vmax=vmax,
                      code=digest({p.name:sha256(p) for p in Path('src').glob('*.py')}))
        cache=Path('runs/image_viewer_epoch17')/digest(identity)[:20]
        ready=(cache/'scores.json').exists() and all((cache/(name+'.png')).exists() for name in ['gt','heatmap','overlay'])
        active=any(j['status'] in ('PENDING','RUNNING') for j in manager.list())
        if st.button('현재 이미지 분석',type='primary',disabled=not valid_checkpoint or active):
            try:
                from src.feature_extractor import FeatureExtractor
                with gpu_lease(),st.spinner('현재 이미지 한 장의 원본·열화 feature를 비교합니다.'):
                    extractor=FeatureExtractor(detector,checkpoint,device,validation)
                    try:out=analyze_image(extractor,record,config,[(ratio,stage,metric)])[(ratio,stage,metric)]
                    finally:extractor.close()
                    views=render(out['original'],out['degraded'],out['values'],out['objects'],out['geometry']['resized_hw'],vmax)
                    views['gt']=draw_gt(views['original'],out['objects'],box_key='input_bbox')
                    save_views({name:views[name] for name in ['gt','heatmap','overlay']},cache)
                    write_json(cache/'scores.json',dict(image_id=record['image_id'],objects=out['objects'],background=out['background'],geometry=out['geometry'],identity=identity))
                ready=True
            except Exception as e:st.error(str(e))
        if active:st.caption('다른 GPU 작업이 끝나면 현재 이미지 분석을 실행할 수 있습니다.')
        elif not ready:st.caption('화살표로 이미지를 고른 뒤 현재 이미지 분석을 누르세요. 계산한 결과는 저장되어 다시 선택하면 바로 표시됩니다.')
        legend=' &nbsp; '.join(f'<span style="color:{color}">■</span> {name}' for name,color in SIZE_COLORS.items() if name!='underflow')
        st.markdown(legend,unsafe_allow_html=True)
        a,b,c=st.columns(3)
        if ready:
            a.image(str(cache/'gt.png'),caption='원본 + 크기별 GT')
            b.image(str(cache/'heatmap.png'),caption='Heatmap')
            c.image(str(cache/'overlay.png'),caption='원본 + Heatmap overlay')
            b.image(color_scale(vmax))
            b.caption(f'0 (차이 작음) ← {metric} → {vmax:g} (차이 큼)')
            scores=read_json(cache/'scores.json')
            rows=summary_rows(scores['objects'],scores['background'])
        else:
            with Image.open(record['path']) as image:
                a.image(draw_gt(image.convert('RGB'),record['objects']),caption='원본 + 크기별 GT')
            b.info('분석 후 Heatmap 표시')
            c.info('분석 후 Overlay 표시')
            rows=summary_rows(record['objects'],None)
        st.subheader('현재 이미지의 크기별 수치')
        frame=pd.DataFrame(rows).rename(columns={'region':'크기 구간','count':'객체 수','mean':'평균 discrepancy'})
        frame['객체 수']=pd.array(frame['객체 수'],dtype='Int64')
        st.dataframe(frame,hide_index=True,width='stretch',column_config={'평균 discrepancy':st.column_config.NumberColumn(format='%.5f')})
        st.caption('BG: 이 이미지의 GT box 밖 유효 영역 평균. 객체: bbox 내부 평균을 구한 뒤 크기별로 동일 가중 집계. 계산 전/빈 구간은 값 없음.')
        if any(o['size_bin']=='underflow' for o in record['objects']):
            st.caption('2px 미만 객체는 회색 박스로 별도 표시하며 VT에 합치지 않습니다.')

with training_tab:
    st.metric('Best epoch',read_json(ROOT/'pipeline.json',{}).get('selected_epoch',17))
