# CICS

Faster R-CNN R50-FPN의 원본·축소 후 복원 이미지에서 backbone C3/C4/C5 특징 차이를 비교한다. Cosine·L1·L2의 특징 변화는 정보손실률을 의미하지 않는다.

## 폴더

```text
experiments/   실험 코드(src), 설정(configs), Python 실행 코드(cli), 테스트
gui/           GUI 화면·보조 코드·테스트·개별 이미지 캐시(cache)
scripts/       .sh 실행 파일
environments/  의존성·환경 기록·MMCV wheel
results/       학습 가중치(trained), 분석 원본(analysis), 비교 요약(reports)
data/          COCO 사전학습 가중치
.venv/         로컬 가상환경
```

## 환경 및 실행

프로젝트 루트에서 실행한다.

```bash
bash scripts/setup_environment.sh
./scripts/run.sh
./scripts/run_analysis_epoch17.sh
./scripts/evaluate_size_bins.sh
```

GUI 기본 주소는 `0.0.0.0:8501`이며 `CICS_HOST`, `CICS_PORT`로 변경할 수 있다.

새 재현 학습은 `./scripts/run_r50_reproduction.sh`로 실행하며 작업 폴더는 `results/runs/r50_ms20_training/`이다. 기존 epoch별 checkpoint·optimizer 기록은 정리했으므로 과거 학습을 재개할 수는 없다.

## 현재 모델·데이터

COCO 사전학습 가중치를 VisDrone으로 학습한 epoch 17을 사용한다.

- 가중치: `results/trained/faster_rcnn_r50_fpn_coco2visdrone/best_epoch_17.pth`
- 모델 설정: `experiments/configs/epoch17_detector.py`
- 로딩 검증 정보: `experiments/epoch17_validation.json`
- 이미지: `/mnt/icelab_4tb/datasets/VisDrone/images/{train,val,test}`
- 특징 분석 GT: `/mnt/icelab_4tb/datasets/VisDrone/labels/{train,val,test}`
- 학습·평가 GT: `/mnt/icelab_4tb/datasets/VisDrone/annotations_VisDrone_{train,val,dev}.json` (`dev` = test)

학습 중 validation과 best 선택용 test 성능은 각각 `val_metrics.csv`, `test_metrics.csv`에 남아 있다. epoch 17은 보존됐던 epoch 4~20 중 test AP 최고로 선택했다. test가 모델 선택에도 사용됐으므로 독립 holdout 결과로 해석하지 않는다.

분석·test 평가 해상도는 짧은 변 800, 긴 변 최대 1333, 종횡비 유지다. 모델은 10개 VisDrone 클래스를 사용한다. 새 JSON의 category 0·11 및 유효하지 않은 박스는 현재 학습 클래스·박스 필터에서 제외된다.

## 결과

- 전체 분석 원본: `results/analysis/`
- C3 메트릭 비교: `results/reports/metric_comparison_epoch17/`
- stage 비교: `results/reports/stage_comparison_epoch17/`
- best test 재평가: `results/trained/faster_rcnn_r50_fpn_coco2visdrone/best_test_size_metrics.csv`
- 평가 조건·해시: 같은 폴더의 `evaluation_test.json`

VT [2,8), T [8,16), S [16,32), M [32,64), L [64,∞)는 원본 GT의 `sqrt(width*height)` 기준이다. 기존 COCO S/M/L 구간과 다르다. AP·AP50·AP75·AR의 CSV 원시값은 0~1이며, 퍼센트는 100을 곱한다. AP·AR은 IoU .50:.95, 최대 탐지 500 기준이다. 과거 전체 epoch의 VT/T 성능은 미기록이며 현재는 best 17을 평가했다. 새 학습은 크기별 성능을 epoch마다 함께 기록한다.

## 검증·Git

```bash
.venv/bin/python -m pytest -q
```

코드·설정·테스트·의존성을 버전 관리한다. 가중치·결과·GUI 캐시·venv·wheel·로그·로컬 비밀 설정은 `.gitignore`로 제외한다. 환경 정보는 `experiments/cli/check_environment.py`로 `environments/environment.json`에 다시 기록할 수 있다. 초기 smoke·검사·논문/가중치 조사 자료는 정리했다.
