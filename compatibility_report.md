# Compatibility report

> 추가 조사 (2026-10-07): 공개 VisDrone checkpoint 2개를 실제 확보했다.
> VLM4RSDet의 R50-FPN 배포본은 현재 모델의 348개 tensor와 strict 일치하고
> RTX 5090 C3/C4/C5 추출에 성공했다. 전체 평가와 provenance 불일치 검증은 남아 있다.
> 최신 결과: [checkpoint_search/RESULTS.md](reports/checkpoint_search/RESULTS.md).
> 아래 초기 조사의 “확보하지 못함”은 추가 조사 이전 상태다.

검사일: 2026-10-07 (Asia/Seoul). 이 문서는 실행 결과와 미검증 항목을 구분한다.

## 확인한 환경

- GPU: NVIDIA RTX 5090, 32 GB, driver 595.91.07. GPU tensor 연산 성공.
- Python 3.11.14, PyTorch 2.9.1+cu128, torchvision 0.24.1+cu128.
- 시작 시 MMEngine/MMCV/MMDetection, scipy/pandas/matplotlib/Streamlit 미설치.
- 기존 MMDetection commit: cfd5d3a985b0249de009b67d04f37263e11cdf3d.
- SR-TOD commit: 1c4d93b104450055ecbfed178e6ffbbf1aedc9f6.
- SR-TOD의 문서상 torch 1.12 / mmcv 2.0.1 환경을 Blackwell GPU에 그대로 적용하지 않는다.

## 데이터 및 baseline

`/mnt/icelab_4tb/datasets/VisDrone/{images,labels}/{train,val,test}`에 YOLO 라벨이 있다.
인접 프로젝트 변환 코드를 확인: 0–9 클래스, ignore/others 정보가 제거된 export다.
따라서 현재 데이터의 BG는 **알려진 GT box 외부**이며, 원본 ignore 영역 제거를 보장할 수 없다.
원본 자료는 수정하지 않는다. 별도 COCO 변환에 source file/line ID를 보존한다.
`VisDroneDataset/*_remove_SO`는 객체 제거 영상이므로 분석 입력으로 사용하지 않는다.
전체 audit 결과는 `reports/dataset_audit.json`에 기록한다.

기존 standard Faster R-CNN R50-FPN의 RGB mean/std와 divisor=32를 확인했다.
분석은 종횡비 유지 long-edge 640, 우하단 padding, RGB FP32로 고정한다.
SR-TOD 전용 reconstruction 모듈은 포함하지 않는다. C3/C4/C5는 backbone.layer2/3/4다.

## 공개 checkpoint 조사

- https://github.com/Hiyuur/SR-TOD : 공식 코드, Apache-2.0. 로컬 README에는 검증 가능한 standard baseline weight 배포가 없다.
- https://github.com/oulutan/Drone_FasterRCNN : MIT, VisDrone weight 링크는 있으나 legacy maskrcnn-benchmark 기반이다. 본 MMDetection R50-FPN과의 architecture/key/class/preprocessing/evaluation 호환성은 입증되지 않았다. 채택하지 않았다.
- https://github.com/amiman/VisDrone : torchvision 기반 toolkit. 본 MMDetection baseline의 검증된 checkpoint로 간주하지 않는다.

검증된 VisDrone checkpoint가 아직 없으므로 기본 경로는 COCO pretrained → 60 epoch fine-tuning이다.
공개 checkpoint를 추가할 때 출처/license/commit/train split/class mapping/preprocessing을 provenance JSON으로 제공하고 strict key/shape 검사와 실제 validation을 통과해야 한다.
보고된 성능과 640 실험 성능은 별개로 기록한다. 학습 split 미상은 leakage unknown으로 기록한다.

## 분석 규약

- seed=128. 원본 sqrt(w*h)로 underflow/VT/T/S/M/L 구분.
- bin 경계는 [0,2), [2,8), [8,16), [16,32), [32,64), [64,inf); 0 이하 invalid.
- bilinear antialias downsampling → bilinear upsampling, align_corners=False, round-half-up.
- scalar discrepancy를 input 좌표로 보간한 뒤 fractional pixel area pooling. invalid feature 보간 footprint는 제외.
- GT union과 ignore가 닿는 pixel은 BG에서 보수적으로 제외. 객체 score에서도 ignore가 닿는 pixel 제외.
- cosine norm <=1e-8 invalid; L2는 channel RMS. raw score에는 min-max 정규화 없음.
- 객체 동일 가중, global BG 위치 가중, BG boxplot 이미지 단위. std는 population std.
- val에서 A/B/C 수동 선택; GT-test에 고정 조건 보고. 미확인 test의 독립성은 주장하지 않는다.

## 공식 AI-TOD evaluator 대조

https://github.com/jwwangchn/cocoapi-aitod/blob/master/aitodpycocotools/aitodpycocotools/cocoeval.py
를 직접 확인했다 (`reports/aitod_cocoeval.reference.py`). 해당 evaluator는 VT=[0²,8²],
T=[8²,16²], S=[16²,32²], M=[32²,1e5²]이고 양 끝 경계를 포함한다.
따라서 본 분석의 underflow 분리, 겹치지 않는 반개구간, M/L 64px 분리는
**AI-TOD-style 분석 규약**이다. 공식 evaluator와 완전히 같은 bin이라고 부르지 않는다.

## 전체 데이터 audit 실행 결과

train 6,471장 / 343,204 유효 box, val 548장 / 38,759 box, test 1,610장 / 75,102 box.
모든 이미지에 라벨 파일이 있고 손상 이미지 없음. train의 `9999985_00000_d_0000020:11`은
높이가 0이므로 제외하고 `data/train.json.excluded.json`에 원본 line과 사유를 저장했다.
동일 SHA-256 중복 3쌍(train 내 2, val 내 1), split 간 동일 hash 중복 없음.
중복은 원본 split을 보존하며 장면 단위 독립성은 별도 검증하지 않았다.

## 실행 환경 확정

전용 `.venv`: MMEngine 0.10.7, MMCV 2.1.0, MMDetection 3.3.0.
CUDA 12.8, TORCH_CUDA_ARCH_LIST=12.0, MAX_JOBS=4로 full MMCV wheel을 빌드했다.
CUDA NMS 및 RoIAlign forward/backward 성공. 실제 detector train/val 각 2장,
1 epoch 학습/평가와 validation mAP 기준 best 및 last 저장 성공.
이는 본 학습이 아니며 60 epoch / 전체 split 분석을 완료한 것으로 취급하지 않는다.
전처리 차이: MMDetection 학습/evaluation의 Resize는 기본 OpenCV bilinear,
분석은 명시적인 PyTorch bilinear+antialias이다. 두 전처리를 동일하다고 주장하지 않는다.
현재 학습 기본값은 batch=2, workers=4, AMP off; 실제 smoke는 workers=0으로 실행했다.

CUDA 결정성 오류는 CUBLAS_WORKSPACE_CONFIG=:4096:8을 CUDA context 전에 설정하여 해결했다.
MMEngine HistoryBuffer의 PyTorch 2.9 weights-only 로딩 문제는 알려진 타입만 허용하는
제한된 metadata loader로 해결한다.

공식 COCO 초기 weight SHA-256: `047c8118fc5ca88ba5ae1fab72f2cd6b070501fe3af2f3cba5cfa9a89b44b03e`.
AI-TOD evaluator 참고 파일 SHA-256: `2c3f84612f448dccf085c72f20711f5fb8dcac3022e8f24cdf04af283a24fa1e`.

## 최종 구현 검증

- 단위/통합/GUI 테스트 21개 통과.
- 실제 checkpoint strict key/shape 검사 및 평가 성공.
- epoch 1 checkpoint에서 epoch 2로 실제 재개 성공.
- 실제 detector 7조건 × 2장 분석 export 및 BN buffer 불변 확인.
- 독립 worker process의 분석 완료와 WAITING_FOR_SELECTION 확인.
- GUI 초기/저장 결과 렌더링, 재실행 시 중복 launch 없음, 수동 선택 전 B 실행 차단 확인.
- 60 epoch 본 학습, 전체 split ablation 및 최종 test 보고는 아직 수행하지 않음.

전체 구현 검증 요약: `reports/implementation_validation.json`.

초기 smoke 산출물·로그는 Git 준비 과정에서 정리했다. 위 항목은 당시 검증 기록이며, `scripts/smoke_detector.py`, `scripts/check_analysis_epoch17.py` 및 `tests/`는 재검증을 위해 유지한다.
