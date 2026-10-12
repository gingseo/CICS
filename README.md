# CICS — 소형 객체 정보 손실 민감도 분석

`EXPERIMENT_PLAN.md`의 실험 코드와 Streamlit GUI를 구현했다. 표준 **MMDetection Faster R-CNN + ResNet-50 + FPN**에서 같은 checkpoint의 원본/열화 backbone C3/C4/C5를 비교한다. detector를 ablation마다 다시 학습하지 않는다.

전용 `.venv`에서 실행한다. 전체 데이터 audit, COCO 변환, CUDA custom op 및 detector 학습·평가·저장 통합 검증을 수행했다. 초기 smoke 산출물은 정리했으며 검증 스크립트는 유지한다. 현재 epoch 17 모델을 기준으로 고해상도 test 분석을 진행한다. 사전 검사 및 한계는 [compatibility_report.md](compatibility_report.md)를 참고한다.

## GUI 실행

```bash
cd /mnt/icelab_4tb/gseo/CICS
./run.sh
```

`run.sh`는 실행 위치와 관계없이 프로젝트 `.venv`를 활성화하고 해당 Python으로 GUI와 worker를 실행한다. 가상환경이 없으면 설치 명령을 안내한다. 포트를 바꾸려면 `CICS_PORT=8502 ./run.sh`, 주소를 지정하려면 `CICS_HOST=127.0.0.1 ./run.sh`를 사용한다. 추가 Streamlit 옵션도 인자로 전달할 수 있다. 종료는 `Ctrl+C`다.

서버의 8501 포트로 접속한다. SSH 환경에서는 `ssh -L 8501:localhost:8501 <서버>`로 포워딩한 뒤 `http://localhost:8501`을 연다. GUI는 학습·검증·분석을 별도 worker로 시작하며 새로고침으로 중복 실행하지 않는다. 실행 상태와 로그는 3초마다 갱신한다.

- **실행 / 수동 선택**: checkpoint 검증, 60 epoch 학습, A/B/C/final 실행, 근거 run·선택 이유 저장.
- **실시간 모니터**: 실제 loss/epoch/mAP/LR/GPU/VRAM, 분석 처리 수·ETA, 실패 사유·로그, 중단.
- **Dataset 분석**: raw CSV 기반 표, bin bar/boxplot, 크기–score scatter, 상관·보조 검정, 조건/checkpoint 비교, CSV/PDF/SVG 다운로드.
- **Image viewer**: 저장한 qualitative 결과 또는 사용자 폴더, 이전/다음, 원본/열화/heatmap/overlay/GT score. GT 없는 이미지는 heatmap만 계산한다. 학습·분석 worker가 실행 중이면 추가 GPU 추론은 기다린다.

Heatmap의 기본 고정 범위는 cosine `[0,2]`, L1/L2 `[0,1]`이다. L1/L2가 1을 초과하면 **표시만** 포화될 수 있으므로 viewer의 표시 최대값을 조절한다. raw 통계는 변하지 않는다.

## 서버에서 확인한 데이터

원본 경로: `/mnt/icelab_4tb/datasets/VisDrone/{images,labels}/{train,val,test}`.

| split | 이미지/라벨 파일 | 유효 객체 |
|---|---:|---:|
| train | 6,471 | 343,204 |
| val | 548 | 38,759 |
| test | 1,610 | 75,102 |

손상 이미지·라벨 누락 없음. 높이가 0인 train box 1건은 원본을 보존하고 `data/train.json.excluded.json`에 제외 사유를 저장했다. SHA-256 기준 중복 이미지 3쌍(train 내 2쌍, val 내 1쌍)을 기록하고 원래 split에서 유지한다. split 사이의 동일 파일 hash 중복은 발견되지 않았지만 장면 독립성까지 증명하지는 않는다.

이 데이터는 **ignore/others 정보가 제거된 YOLO export**다. BG는 알려진 GT 바깥이며, 원본 ignore 영역을 완전히 제외했다는 주장을 할 수 없다. 원본 VisDrone DET text 형식도 audit/분석에서 `format: visdrone`으로 지원한다. 원본 ignore annotation의 COCO 변환은 sidecar에 보존되지만 학습/evaluation에서 클래스 독립 ignore 처리는 별도 adapter가 필요하다. 현재 학습 config는 이 서버의 YOLO export만을 대상으로 한다.

```bash
.venv/bin/python scripts/audit_dataset.py --convert --exclude-invalid
```

`reports/dataset_audit.json`에 파일 개수, GT 누락, 손상, annotation 오류, 클래스/bin 분포, 중복, 데이터 fingerprint가 기록된다. 원본 이미지/라벨은 수정하지 않는다. 변환 ID는 `data/*.source_ids.json`에서 원본 파일과 line ID로 추적한다.

## Checkpoint와 학습

검증된 공개 VisDrone checkpoint를 확보하지 못했으므로 **COCO pretrained → 60 epoch fine-tuning**을 기본 경로로 준비했다. 공개 weight를 쓰려면 대응 config와 provenance가 필요하며, strict key/shape 검사 및 실제 COCO bbox evaluation을 통과해야 한다. 임의 weight를 자동으로 유효하다고 표시하지 않는다.

### 1. 학습

```bash
.venv/bin/python scripts/train_detector.py
# 필요 시 AMP를 명시한다. 기본 FP32 경로는 실제 smoke test를 통과했다.
# .venv/bin/python scripts/train_detector.py --amp
```

명령은 별도 worker를 시작하고 job ID를 반환한다. 로그는 `runs/jobs/<job_id>/run.log`, 학습 기록과 best/last checkpoint는 `runs/training/`에 생성된다. `last_checkpoint` 텍스트 파일은 마지막 완료 epoch 파일을 가리킨다. 중단 후 같은 옵션으로 `--resume`을 지정한다. 완료 epoch와 60 epoch 완료 여부를 별도로 기록한다.

확정한 기본값: 1 GPU, batch=2, workers=4, FP32, seed=128, SGD momentum .9, weight decay .0001. LR은 원래 batch 16의 .02를 batch 2에 맞춰 **.0025**로 조정했다. 12 epoch의 8/11 milestone을 60 epoch에 맞춰 **40/55**로 이동했고, warmup은 500 iteration이다. 640 aspect-preserving resize와 수평 반전을 사용한다. full COCO detector weight를 초기화에 사용하며, 80→10 클래스 변경에 따른 classifier/regressor 4개 tensor의 크기 차이는 로그에 드러낸다. SR-TOD 추가 모듈은 사용하지 않는다.

`best`는 매 epoch **validation `coco/bbox_mAP` @[.5:.95]** 최대값이다. COCO API의 전체 AP는 maxDets=100, 모델 출력 상한은 500이며 다른 세부 지표는 evaluator의 maxDets 설정을 따른다. 원본 VisDrone challenge evaluator와 동일한 점수라고 주장하지 않는다.

### 2. 검증

```bash
.venv/bin/python scripts/validate_checkpoint.py \
  --config configs/detector.py \
  --checkpoint runs/training/best_coco_bbox_mAP_epoch_N.pth \
  --provenance runs/training/provenance.json \
  --output runs/best_validation.json
```

`N`은 실제 저장된 epoch 번호로 바꾼다. last도 사용하려면 실제 마지막 epoch 파일에 대해 별도 validation report를 만든다. 외부 checkpoint는 `configs/checkpoint_provenance.example.json`을 복사하여 출처·license·commit·class order·학습 split·원래 전처리를 채운다. 템플릿 placeholder는 실행 시 거부한다. 학습 split 미상은 `unknown`으로 명시한다. `--structure-only`는 구조 검사만 하며 분석 허가로 사용되지 않는다.

validation config는 checkpoint 원래 평가 파이프라인을 담아야 한다. 640 분석 전처리는 별도 manifest에 기록된다. 두 조건의 성능을 동일시하지 않는다. public weight 한 개면 실제 weight 한 개만 표시하며 가상의 best/last를 생성하지 않는다.

## 순차 분석

```bash
.venv/bin/python scripts/run_analysis.py \
  --checkpoint runs/training/best_coco_bbox_mAP_epoch_N.pth \
  --validation runs/best_validation.json --phase A --split val
```

A는 C3/cosine에서 r=.75/.5/.25를 비교한다. GUI에서 완료 결과를 확인하고 이유와 함께 ratio를 선택한다. 이후 동일 명령의 `--phase B`, stage 선택 후 `--phase C`를 실행한다. CLI 선택도 지원한다.

```bash
.venv/bin/python scripts/select_condition.py --phase A --value 0.5 \
  --reason '분포와 이미지 간 일관성을 검토한 실제 선택 이유' \
  --runs runs/<A_run1> runs/<A_run2> runs/<A_run3>
```

B는 선택 ratio에서 C3/C4/C5, C는 선택 stage에서 cosine/L1/L2를 비교한다. 근거 run이 완료되어야 하고 같은 checkpoint와 dataset을 사용해야 한다. 상위 선택을 바꾸면 하위 선택을 무효화하고 이력은 보존한다. 다음 조건을 자동으로 고르지 않는다.

최종 metric까지 선택한 뒤 `--phase final --split test`로 고정 설정을 보고한다. test GT가 없거나 일부 누락되면 complete-GT val로 fallback하며 이유를 manifest/GUI에 표시한다. best/last 비교는 같은 selection으로 각각 final을 실행한다. 선택에 사용한 checkpoint와 비교 checkpoint의 차이는 manifest에 남는다. `--limit 3`은 소규모 검사, 생략은 전체 split이다. subset 결과와 전체 split 결과는 서로 다른 run ID를 갖는다.

## 계산 규약과 저장물

- 원본 annotation `sqrt(w*h)`로 `underflow (0,2), VT [2,8), T [8,16), S [16,32), M [32,64), L [64,∞)`를 분류한다. 공식 AI-TOD evaluator와 동일한 평가 bin이 아니라 명시적인 **AI-TOD-style 분석 bin**이다.
- long edge 640, aspect ratio 유지, 우하단 divisor-32 padding. resize와 degradation은 FP32 bilinear, downsample antialias, align_corners=False, 크기는 round-half-up. 열화에는 유효 영상만 사용한 뒤 normalization/padding을 적용한다. 실제 중간 H/W를 저장한다.
- C3/C4/C5는 backbone.layer2/layer3/layer4, 512/1024/2048 channels, stride 8/16/32. FPN 출력이 아니다.
- cosine은 채널 축 `1-cos`, L1은 채널 평균 절댓값, L2는 채널 RMS. cosine의 norm ≤1e-8은 invalid이며 발생 비율을 이미지별로 남긴다.
- scalar map을 network-input 좌표로 bilinear 보간하고 box–pixel 교차 면적으로 가중 평균한다. invalid feature를 보간한 footprint는 제외한다. 작은 box도 score를 정의하지만 원래 feature 공간 해상도가 증가하는 것은 아니다.
- 객체 bin은 객체당 동일 가중. BG는 유효 pixel 단위 global 평균과 이미지별 BG 평균 분포를 분리한다. GT/ignore가 조금이라도 닿는 pixel은 BG에서 보수적으로 제외하고 padding도 제외한다. 객체에서도 ignore pixel을 제외한다.
- raw score는 min-max 정규화하지 않는다. 빈 bin은 NA/count=0. std는 population std이며 신뢰구간이 아니다. boxplot은 1.5 IQR whisker, outlier는 표시만 생략한다. scatter만 최대 20,000개 seed=128로 표시하고 전체 통계는 모든 객체를 사용한다.
- Spearman/Pearson, Kruskal–Wallis, bin 대 BG Mann–Whitney U 및 Holm 보정을 저장한다. 객체의 이미지 내 군집성과 BG/객체 sample 단위 차이 때문에 p-value는 보조 결과다. image-cluster bootstrap CI는 현재 제공하지 않는다.

`runs/<content_id>/`에는 `config.yaml`, `manifest.json`, `state.json`, `objects.csv`, `background.csv`, `summary.csv/json`, `statistics.json`, `figures/*.{png,pdf,svg}`, 이미지별 `shards/`, `qualitative/`가 저장된다. 기본 qualitative export는 처음 12장이고 그 외 이미지는 viewer에서 추론한다. manifest는 seed, 환경, 데이터/weight/config/code 식별값, 조건, 모듈 shape, pooling/제외 정책, split 용도·fallback·선택 이력을 기록한다.

결과 재사용은 데이터 내용, checkpoint, 전처리/연산 설정, 소스 코드, ratio/stage/metric으로 식별한다. 같은 image의 feature pair를 여러 metric/stage가 공유한다. 전체 feature cache는 저장하지 않는다. 중단된 job은 완료 이미지 shard부터 재개한다. 오류는 FAILED/STOPPED로 남으며 DONE으로 처리하지 않는다. 소스 변경으로 fingerprint가 바뀌면 새 run을 생성한다.

## 환경 재현 및 검증

현재 조합: Python 3.11.14, PyTorch 2.9.1+cu128, torchvision 0.24.1+cu128, MMEngine 0.10.7, MMCV 2.1.0, MMDetection 3.3.0. CUDA 12.8 nvcc로 **sm_120** 전용 MMCV wheel을 직접 빌드했다. 이 wheel은 다른 CUDA/Python/PyTorch/GPU 조합의 호환성을 보장하지 않는다.

```bash
# 같은 서버 조합에서 재설치
bash scripts/setup_environment.sh
.venv/bin/python scripts/check_environment.py
.venv/bin/python -m pytest -q
# 실제 GPU 학습·평가 통합 검사 (2장, 1 epoch; reports/에만 저장)
.venv/bin/python scripts/smoke_detector.py
```

`reports/implementation_validation.json`에 테스트 21개, GPU 학습·재개·평가·분석·GUI 검증을 요약했다.

`reports/environment.json`, `reports/environment.freeze.txt`, `reports/mmcv_build.log`에 로컬 환경 기록을 남겼다. 초기 smoke 로그는 정리했다. 빌드 wheel은 `reports/wheels/`에 있다. PyTorch는 base 환경을 읽어 사용하며 다른 환경을 변경하지 않았다. `CUBLAS_WORKSPACE_CONFIG=:4096:8`, cuDNN deterministic, seed=128을 적용한다. 분석의 deterministic 경고는 worker 로그에 보존한다.

MMEngine checkpoint는 PyTorch `weights_only=True`로 읽으며, HistoryBuffer와 제한된 NumPy 메타데이터만 명시적으로 허용한다. 임의 pickle 로딩으로 fallback하지 않는다. 학습 중단/재개는 완료 epoch checkpoint를 사용하며 현재 minibatch 중간 상태는 복구하지 않는다.

### 고해상도 Faster R-CNN 재현 실험

논문 수치와 기존 640 입력 baseline 차이는 `reports/reproduction/FINDINGS.md`에 기록했다.
고해상도 20 epoch 학습 → 보존된 epoch별 test 평가 → test AP 최고 checkpoint 선택 → test 재평가 및 그래프 생성:

```bash
./run_r50_reproduction.sh
```

이미 실행 중이면 중복 실행하지 않는다. 중단된 pipeline을 이어 진행할 때는
`./run_r50_reproduction.sh --resume`을 사용한다. 출력은 `runs/reproduction_r50_ms20/`에 저장된다.
`pipeline.json`은 전체 진행 및 test 기준 선택 결과, `training_status.json`은 실시간 학습 상태다.
기존 `runs/training/`의 640 baseline은 별도로 보존된다.

독립 실행 명령:

```bash
source .venv/bin/activate
python scripts/train_detector.py --detector-config configs/detector_r50_visdrone_ms20.py --work-dir runs/reproduction_r50_ms20
```

독립 명령은 학습만 실행한다. 최종 평가와 그래프 자동 생성은 위 shell pipeline이 담당한다.
새 실험은 표준 R50-FPN에 공개 VisDrone 다중 스케일 recipe를 적용한 실험이며,
논문의 Faster R-CNN checkpoint를 확보한 정확 재현은 아니다.

현재 실행은 사용자 요청에 따라 test로 checkpoint를 선택하고 같은 test로 결과를 보고한다(`independent_holdout=false`). 전환 전에 삭제된 1–3 epoch은 후보에서 제외되며, 4 epoch부터 남아 있는 checkpoint를 보존해 비교한다. 학습 중 val 그래프는 진단용으로 유지되고, 최종 선택은 test AP가 결정한다. `test_selection/epoch_metrics.csv`와 `test_selection/curves.png`에 test 기준 비교, `eval_test/report.json`에 선택된 모델의 재평가 결과가 저장된다.

### 17 epoch 모델로 feature discrepancy 분석

GUI 기본 화면은 `epoch_17.pth`, test 1610장, detector와 일치하는 OpenCV
bilinear Resize(짧은 변 800 / 긴 변 최대 1333, 비율 유지)로 설정되어 있다.
이는 학습의 다중 스케일을 평가할 때 쓰는 고정 전처리이며, 학습 전체가 단일 해상도였다는 의미는 아니다.
기존 640 분석 설정은 `configs/analysis.yaml`, 현재 설정은 `configs/analysis_epoch17.yaml`로 구분한다.
열화 연산은 유효 영역에서 bilinear+antialias 축소 후 bilinear 복원하며 normalization/padding 전에 적용한다.

```bash
CICS_HOST=127.0.0.1 ./run.sh         # GUI: 서버 내부 포트 8501
./run_analysis_epoch17.sh           # A: r=.75/.50/.25, C3, cosine, 전체 test
./run_analysis_epoch17.sh --resume  # 중단한 동일 작업 재개
```

GUI의 `실험 진행`에서 진행률과 A/B/C 상태, `조건별 통계`에서 크기별 비교 표·그래프,
`이미지 비교`에서 원본·열화·heatmap·GT를 볼 수 있다. 분석 중에도 먼저 저장된 12장의 이미지는 볼 수 있다.
전체 작업 완료 시 결과가 갱신된다. 필요하면 상단 `결과 새로고침`을 누른다.
학습 기록과 test checkpoint 선택 표는 `학습 기록`에 따로 표시된다.

A 완료 후 GUI에서 근거와 함께 축소 비율을 선택해야 B 실행이 가능하고,
B 완료 후 stage를 선택해야 C 실행이 가능하다. 다음 단계 조건은 자동 선택하지 않는다.
결과: `runs/analysis_epoch17_highres/<run_id>/`; 선택 이력: 같은 폴더의 `selection.json`.
통계의 BG는 이미지 단위, 객체 bin은 객체 단위이며 global BG 평균은 픽셀 가중으로 별도 표시한다.
기본 heatmap scale은 metric별로 고정하고 raw discrepancy에 이미지별 min-max 정규화를 적용하지 않는다.

검증: detector Resize와 가로·세로·정사각형 이미지의 픽셀/scale factor 일치,
선택 모델의 동일 입력 discrepancy 최대 약 2.24e-7, BN buffer 불변,
2장×A 세 조건 GPU smoke와 populated GUI 검사, 전체 24개 테스트 통과.
초기 smoke 결과는 정리했다. 필요하면 `scripts/check_analysis_epoch17.py`로 다시 생성할 수 있다.
서버가 원격이면 SSH 포트 포워딩으로 로컬 브라우저에서 내부 GUI에 접속할 수 있다.

### 간소화한 이미지 비교 화면

이미지 비교가 첫 탭이며, 원본 GT 기준 VT+T 객체 비율이 50%를 초과하는 이미지만 탐색한다.
현재 test 후보는 350장이고 VT+T 객체 수 내림차순이다. 목록은 `reports/viewer_vt_t_test.csv`에 기록했다.
화살표로 이동 후 `현재 이미지 분석`을 누르면 현재 한 장·선택한 조건만 계산한다.
원본+GT, heatmap, overlay를 나란히 표시하고 GT는 VT/T/S/M/L별 색상 및 범례를 제공한다.
아래에는 BG/VT/T/S/M/L의 객체 수와 평균 discrepancy만 표시한다. 객체별 긴 상세 표는 표시하지 않는다.
계산 결과는 `runs/image_viewer_epoch17/`에 별도 저장한다. 이미지·라벨·가중치·조건이 다르면 기존 결과를 재사용하지 않는다.

학습 기록 탭에는 선택된 best epoch(17)만 표시한다. 로그·학습 곡선·전체 epoch 표는 이 탭에서 제거했다.
A/B/C 조건 선택에 사용자 이유 입력은 요구하지 않는다.

`실험 진행 → 전체 데이터 통계 계산`은 이미지 뷰어와 별도 기능이다.
N=0이면 전체 split, N>0이면 파일명 순서의 앞 N장에 대해 단계별 통계를 계산한다.
이는 detector 재학습이 아니며 가중치를 변경하지 않는다. VT/T 뷰어 필터가 전체 통계에 자동 적용되지는 않는다.
화면 변경 후 실행 중인 GUI를 새로고침하면 된다. GUI 서버의 실행/종료는 사용자가 직접 한다.

### Git 버전 관리

코드, 설정, 테스트, 실행 스크립트, 문서를 버전 관리한다. `.venv/`, `data/`, `runs/`, 자동 생성 `reports/`, 가중치, wheel, 로그, 캐시 및 로컬 비밀 설정은 `.gitignore`로 제외한다. 논문 비교와 가중치 조사 문서인 `reports/reproduction/FINDINGS.md`, `reports/checkpoint_search/RESULTS.md`는 포함한다.

실제 학습·분석 결과와 가중치는 로컬에 보존한다. 초기 smoke 산출물과 이전 GUI 백업은 정리했으며, smoke 검증 코드와 테스트는 유지한다. 다른 서버에서는 환경·데이터·가중치를 별도로 준비해야 한다.
