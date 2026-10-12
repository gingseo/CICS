import pytest
from pathlib import Path


def test_gui_rerun_does_not_start_jobs_or_skip_manual_selection(monkeypatch):
    pytest.importorskip('streamlit')
    from streamlit.testing.v1 import AppTest
    launched=[]
    monkeypatch.setattr('src.job_manager.JobManager.launch',lambda *a,**kw:launched.append(a))
    app=AppTest.from_file(Path(__file__).resolve().parents[1]/'app/streamlit_app.py',default_timeout=30).run()
    assert not app.exception
    next(x for x in app.selectbox if x.label=='단계').select('B')
    app.run()
    assert not app.exception and not launched
    assert next(x for x in app.button if x.label=='전체 통계 계산 시작 / 재개').disabled


def test_simple_viewer_navigation_and_no_training_logs(monkeypatch):
    from streamlit.testing.v1 import AppTest
    launched=[]
    monkeypatch.setattr('src.job_manager.JobManager.launch',lambda *a,**kw:launched.append(a))
    app=AppTest.from_file(Path(__file__).resolve().parents[1]/'app/streamlit_app.py',default_timeout=30).run()
    assert not app.exception
    assert not app.text_area
    assert not any(x.label in ['폴더 이미지','저장된 이미지'] for x in app.selectbox)
    before=app.session_state['tiny_viewer_index']
    next(x for x in app.button if x.label=='다음 →').click();app.run()
    assert not app.exception and app.session_state['tiny_viewer_index']==before+1
    next(x for x in app.button if x.label=='← 이전').click();app.run()
    assert not app.exception and app.session_state['tiny_viewer_index']==before
    assert not launched
    training=next(t for t in app.tabs if t.label=='학습 기록')
    assert len(training.metric)==1 and training.metric[0].value=='17'
    assert not training.dataframe and not training.json
