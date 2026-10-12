# 소형 객체 정보 손실 민감도 분석 — 실험 및 GUI 구현 명세

- 작성일: 2026-10-07
- 원문: https://chatgpt.com/share/6ac5abb3-0958-83ee-82c2-ac4e084fb8ec
- 목적: 공유 대화의 최종 결정을 보존하고, 후속 구현자가 실험 코드와 GUI를 구축할 수 있도록 정리함.
- 문서 상태: 실험 계획임. 코드 구현·데이터 검사·공개 checkpoint 조사·실험 실행은 아직 수행하지 않음.
- 해석 원칙: 대화 후반의 변경을 우선함. **확정**, **대화상 제안**, **추가 제안**, **코드 확인 후 결정**을 구분함. 본 문서의 제안을 사용자가 이미 승인한 결정으로 취급하지 않음.

## 1. 연구 목적과 범위

### 1.1 핵심 질문

동일한 검출 모델에 원본 입력과 해상도 열화 입력을 넣었을 때 발생하는 feature discrepancy가, 배경 및 큰 객체보다 작은 객체에서 더 크게 나타나는가?

원본과 열화 입력의 같은 공간 위치 feature vector를 비교하고, GT 객체 크기별로 discrepancy의 평균과 분포를 분석함. 이번 연구의 중심은 검출 성능 향상이 아니라 **정보 손실에 민감한 영역을 찾는 신호로서 discrepancy의 유효성 검증**임.

### 1.2 연구 배경과 주장 수준

기존 대화는 self-reconstruction residual → 객체 제거 target을 사용하는 ICR → feature-level discrepancy 순으로 발전함. 원본 복원을 최적화하면서 그 복원 오차를 guide로 사용하는 구조의 긴장이 출발점임. 다만 복원 학습이 작은 객체의 오차를 반드시 없앤다고 단정하지 않음.

ICR의 residual은 객체 존재 여부에 따른 차이를 포함하므로 순수한 해상도 정보 손실과 동일시하지 않음. 이상 탐지에서는 teacher–student 구조 자체보다 **표현 간 차이를 위치 단서로 활용하는 원리**를 참고함. 작은 객체를 anomaly로 정의하지 않음.

본 실험의 discrepancy는 해상도 열화에 따른 표현 변화량이며, **information-loss sensitivity의 proxy**로 해석함. 정보이론적 손실량, 손실률 또는 실제 객체 정보 소실의 직접 측정값으로 단정하지 않음. 입력 열화 실험만으로 backbone 내부 downsampling의 인과적 효과까지 입증했다고 주장하지 않음.

### 1.3 이번 구현에 포함하지 않는 범위

- discrepancy 기반 feature enhancement와 그에 따른 mAP 개선 모델: 후속 연구.
- teacher–student, reconstruction 또는 inpainting 모듈의 신규 학습: 본 실험에 불필요함.
- cross-stage feature 직접 비교: 현재 C3/C4/C5 ablation과 다름. 각 stage 안에서 원본–열화 feature를 비교함.
- 객체 주변을 추가로 제거하는 pure-background supplementary: 사용자 요청에 따라 제외함.
- 모델별 대규모 비교 및 전체 조합 grid search: 기본 범위에 포함하지 않음.

## 2. 최종 결정과 미확정 항목

| 항목 | 내용 | 상태 |
|---|---|---|
| 검출 모델 | Faster R-CNN + ResNet-50 + FPN | 확정 |
| 데이터 | 보유한 VisDrone 데이터 사용, 실제 구조와 annotation 검사 선행 | 확정 |
| 우선 실행 경로 | VisDrone으로 학습된 공개 weight + 대응 code/config 확보 후 검증 | 최종 방향 |
| 학습 대안 | 적절한 weight가 없으면 COCO pretrained → VisDrone 60 epoch fine-tuning | 확정 |
| 자원 | RTX 5090 1장 | 해당 대화 기준, 실행 서버에서 재확인 |
| 구현 기반 | MMDetection 및 SR-TOD 코드 참고 | 확정, 실제 버전은 검사 후 고정 |
| 입력 | 640 기준, 원본 코드의 resize·padding 정책을 먼저 확인 | 세부 미확정 |
| seed | 128 | 후반 정리 기준 채택. 이전 42는 사용하지 않음 |
| best checkpoint | validation bbox mAP@[0.5:0.95] 기준 | 확정 |
| 분석 split | GT가 있는 test 우선, 없으면 val | 대화 방향, 데이터 검사 필요 |
| 객체 크기 | resize 전 원본 annotation의 sqrt(w×h) | 확정 |
| size bins | AI-TOD-style VT/T/S/M + 확장 L | 방향 확정, 정확한 경계 확인 필요 |
| feature | backbone C3/C4/C5, FPN P3/P4/P5 아님 | 확정 |
| degradation | 640→480→640, 640→320→640, 640→160→640에 해당하는 비율 | 확정, 연산 세부 미확정 |
| metric | cosine 주 지표, L1/L2 비교 | 확정 |
| ablation 순서 | degradation → feature stage → metric | 확정 |
| 단계 간 선택 | 결과 확인 후 GUI에서 수동 선택 | 대화상 권고, 명시적 최종 응답 없음 |
| 객체 집계 | bbox 내부 평균 → 객체를 동일 가중치로 bin별 집계 | 대화상 제안, 본 문서 기본안 |
| BG 집계 | 모든 GT box union 외부의 유효 위치를 전체 데이터에서 합산 | 대화상 제안, 본 문서 기본안 |
| 분포 시각화 | BG / VT / T / S / M / L box plot | 요청됨 |
| 통계 | mean/std/median/count, Spearman 등. 유의성 검정은 보조로 계산 | 대화 방향 |
| checkpoint 비교 | best/last가 있으면 둘 다, 공개 weight 하나면 하나만 표시 | 최종 방향 |
| GUI | 실시간 학습·분석 모니터 + 이미지 viewer + dataset-level 분석 | 확정 |
| GUI 프레임워크 | Streamlit | 대화상 제안 |

**변경 이력:** “반드시 학습”이라는 초기 설명은 “검증 가능한 VisDrone checkpoint가 있으면 학습 생략”으로 수정됨. 분석 split은 초기 val 논의보다 후반 test 우선 결정을 반영함. 크기 분류는 resize 이후가 아니라 원본 annotation 기준임.

## 3. 구현 시작 전 compatibility report

먼저 `compatibility_report.md`를 작성함. 결과 없이 특정 checkpoint가 존재하거나 호환된다고 가정하지 않음.

### 3.1 데이터 검사

- 실제 dataset root와 train/val/test 이미지·annotation 경로를 확인함.
- split별 이미지 수, annotation 수, 대응 누락, 손상 이미지, 중복 이미지 여부를 기록함.
- annotation format, class mapping, 유효하지 않은 box, ignored region 및 평가 제외 대상의 처리 규칙을 확인함.
- 원본 annotation을 보존함. 변환이 필요하면 별도 경로에 생성하고 원본 ID와 변환 ID 대응을 저장함.
- GT가 없는 split에는 객체별 통계를 계산하지 않음. GT 부재를 객체가 없는 이미지로 취급하지 않음.
- test에 GT가 없으면 val로 fallback한 사유를 manifest와 GUI에 표시함.

### 3.2 공개 checkpoint 조사 및 검증

1. VisDrone-trained Faster R-CNN R50-FPN과 대응 config/code가 함께 제공되는지 조사함.
2. 출처 URL, license, 코드 commit, config, checkpoint hash, 학습 split, class mapping, 전처리를 기록함.
3. architecture와 state dict의 대응을 검사함. 누락·예상 밖 key를 숨기지 않음.
4. 대응 평가 파이프라인으로 성능을 확인함. 보고된 값과 비교 가능한 조건인지도 구분함.
5. checkpoint의 학습 데이터에 최종 분석 split이 포함되었는지 확인함. 알 수 없으면 그 한계를 기록함.
6. 검증 실패 또는 적절한 weight 부재 시 COCO pretrained fine-tuning 경로로 전환함.

공개 weight의 원래 전처리와 이번 640 분석 전처리가 다르면, **checkpoint 재현 평가**와 **본 실험 분석 설정**을 별도로 기록함. 둘을 동일 조건의 성능으로 표시하지 않음.

### 3.3 환경 검사

Python, PyTorch, CUDA, GPU driver, MMEngine, MMCV, MMDetection, SR-TOD commit을 기록함. 실제 GPU에서 tensor 연산과 필요한 custom op를 검증함. 특정 버전 조합은 본 문서에서 미리 단정하지 않음.

배치 크기·AMP·worker 수는 자원 검사 후 정함. 실행 가능한 조합과 설치 방법을 환경 파일에 고정함.

## 4. Detector 학습과 checkpoint 관리

### 4.1 공개 checkpoint 경로

`데이터/환경 검사 → checkpoint 검증 → discrepancy 분석`

Training 화면은 `SKIPPED: validated checkpoint`로 표시함. 존재하지 않는 epoch/loss/best/last 기록을 만들지 않음.

### 4.2 Fine-tuning 경로

`COCO pretrained → VisDrone 60 epoch → validation best/last 저장 → discrepancy 분석`

- detector는 기본적으로 한 번 학습함. degradation/stage/metric마다 재학습하지 않음.
- MMDetection/SR-TOD의 확인된 baseline 설정을 출발점으로 사용함. SR-TOD 전용 추가 모듈을 baseline에 자동 포함하지 않음.
- optimizer, LR, schedule, warmup, batch size, augmentation, AMP는 실제 config 검사 후 확정함.
- 60 epoch와 단일 GPU에 맞는 schedule 및 LR 조정 내용을 명시함. 원래 config의 값을 설명 없이 그대로 옮기지 않음.
- 검증과 best 선택에는 train이 아닌 validation split을 사용함.
- `best`는 detection mAP 기준임. discrepancy 결과가 가설에 잘 맞는 checkpoint라는 뜻이 아님.
- last는 마지막으로 완료한 epoch의 checkpoint임. 중단된 run에서는 60 epoch 완료 여부를 별도 표시함.

분석 시 `eval()` 및 gradient 비활성화를 적용함. 원본과 열화 입력은 같은 고정 weight를 사용함. 분석 과정에서 BN 통계나 weight가 변하면 안 됨.

## 5. 입력과 degradation 정의

### 5.1 공통 전처리

640×640 강제 변형 여부는 확정되지 않음. 실제 baseline의 resize와 aspect-ratio 유지, padding, 채널 순서, normalization을 확인한 뒤 하나의 분석 파이프라인으로 고정함.

모든 이미지에서 원본 크기, resize 후 크기, network tensor 크기, scale factor, padding offset, valid-image mask를 저장함. GT는 이 변환을 따라 이동하되 **size bin은 원본 w/h 기준으로 유지**함.

### 5.2 Degradation 기본안 — 추가 제안

기하 변환이 완료된 유효 이미지 영역을 기준으로 downsample 후 원래 공간 크기로 복원함. 원본 branch와 열화 branch는 같은 normalization과 padding을 사용함. padding이 resize에 섞여 인공적인 경계 차이를 만들지 않도록 함.

| 설정 | 축별 축소 비율 r | 정사각형 640 입력일 때 |
|---|---:|---|
| d075 | 0.75 | 640→480→640 |
| d050 | 0.50 | 640→320→640 |
| d025 | 0.25 | 640→160→640 |

비정사각형 유효 영역에는 H/W에 같은 비율을 적용함. 실제 중간 H/W와 반올림 규칙을 기록함. 이때 화면에 실제 크기를 표시하고 무조건 `640→320→640`으로 표기하지 않음.

downsample interpolation, antialias, upsample interpolation, align_corners, 연산 dtype과 삽입 위치는 config로 명시함. **초기 구현 권고**는 bilinear downsample + antialias, bilinear upsample임. 본 실험에서는 해당 정책을 고정하고 비율만 비교함.

## 6. Feature discrepancy 계산

각 stage k에 대해:

\[
F_k^{ori}=f_k(I),\qquad F_k^{deg}=f_k(T_r(I)).
\]

C3/C4/C5가 코드상 어느 module인지 직접 확인하고 module name, channel 수, stride, 실제 tensor shape를 저장함. 같은 stage의 두 branch를 비교하므로 별도의 학습 가능한 feature alignment는 기본 구조에 필요하지 않음.

### 6.1 Metric 정의 — 구현 기본안

위치 p의 feature vector를 a, b, channel 수를 C라 함.

\[
D_{cos}(p)=1-\frac{a^Tb}{\|a\|_2\|b\|_2}
\]

\[
D_{L1}(p)=\frac1C\sum_c|a_c-b_c|,\qquad
D_{L2}(p)=\sqrt{\frac1C\sum_c(a_c-b_c)^2}.
\]

- cosine은 channel 축에서 계산함. 공간 전체를 flatten한 단일 cosine으로 대체하지 않음.
- L2는 본 기본안에서 channel RMS로 정의함. 일반 Euclidean norm과 구분하여 이름·수식을 저장함.
- cosine의 수학적 범위는 일반적으로 [0,2]임. feature 특성에 따른 실제 범위와 혼동하지 않음.
- norm이 epsilon 이하인 vector의 처리는 아직 미확정임. **권고:** invalid로 표시하고 발생 비율을 기록함. 두 zero vector에 discrepancy 1을 부여하는 식의 구현을 피함.
- norm과 metric 계산은 FP32를 권고함.
- raw 수치에 이미지별 min-max normalization을 적용하지 않음. 표시용 normalization은 통계와 분리함.
- L1/L2와 cosine은 단위와 범위가 다름. raw 평균의 크기만으로 metric 우열을 판단하지 않음.

## 7. GT 크기, 객체 score와 BG score

### 7.1 Size bins

\[
s_{original}=\sqrt{w_{original}h_{original}}.
\]

아래는 **AI-TOD-style 구현 후보**임. 공식 evaluator의 정확한 경계 및 포함 규칙은 구현 전에 확인함. 이 표를 공식 평가 코드와 이미 대조한 결과로 취급하지 않음.

| 구간 | 후보 범위 |
|---|---|
| underflow | 0 < s < 2 |
| VT | 2 ≤ s < 8 |
| T | 8 ≤ s < 16 |
| S | 16 ≤ s < 32 |
| M | 32 ≤ s < 64 |
| L | 64 ≤ s |

L은 이 분석을 위한 확장 구간임. underflow가 있으면 별도 count와 결과를 기록하며, VT로 조용히 합치거나 삭제하지 않음. bbox w/h가 0 이하이면 invalid로 기록함.

### 7.2 객체별 평균

\[
d_i=\frac{1}{|B_i|}\sum_{p\in B_i}D(p),\qquad
\mu_g=\frac{1}{N_g}\sum_{i\in g}d_i.
\]

각 객체 score를 먼저 계산한 후 객체당 동일 가중치로 bin 통계를 계산함. 큰 box의 pixel이 더 많다는 이유로 해당 객체의 가중치가 커지지 않게 함.

겹치는 GT는 객체별 독립 집계에 허용함. BG에서는 전체 GT union을 한 번 제외함. bbox 기반 score에는 객체 주변 배경도 포함될 수 있으므로 segmentation 내부의 순수 객체 score처럼 표현하지 않음.

### 7.3 Tiny box의 좌표 처리 — 구현 전 결정 필요

작은 객체가 feature map의 한 cell보다 작으면 단순 정수 반올림이 빈 ROI 또는 과도한 ROI를 만들 수 있음.

**추가 제안:** scalar discrepancy map을 공통 network-input 좌표계로 bilinear interpolation한 뒤, box와 pixel의 겹침 면적을 가중치로 평균함. 유효 mask와 padding 제외 규칙도 동일 좌표계로 적용함. subpixel box에도 정의된 score를 얻을 수 있으나, upsample이 원래 feature의 공간 해상도를 높여 주는 것은 아님.

대안으로 native feature map에 fractional-area pooling을 적용할 수 있음. 어느 방식을 택하든 모든 stage와 metric에서 동일하게 사용하고 방법을 명시함. native cell 수 및 input-space box 크기를 함께 기록해 해상도 한계를 확인함.

### 7.4 Background

BG는 유효 이미지 안에서 모든 GT box union을 제외한 영역임. padding은 제외함. ignored region 및 평가 제외 annotation을 BG에 포함할지는 원본 annotation 규칙 확인 후 정함. **권고:** 알려진 ignore 영역은 객체 통계와 BG 통계 양쪽에서 제외하여 ‘배경’으로 오인하지 않게 함. 이는 사용자가 제외한 pure-background 주변 확장 실험과 별개임.

\[
\mu_{BG}=\frac{\sum_j\sum_{p\in BG_j}D_j(p)}{\sum_j|BG_j|}.
\]

객체 통계는 instance 단위, BG 평균은 유효 위치 단위임. 같은 통계 단위라고 표시하지 않음.

**BG box plot 기본안:** 이미지별 BG 평균을 한 sample로 사용함. 객체 bin은 객체별 평균을 sample로 사용함. 표에는 global BG 평균과 이미지별 BG 평균 분포의 요약을 분리하고 `sample_unit`, `n_images`, `n_objects`, `n_valid_locations`를 저장함.

## 8. 순차 Ablation

| 단계 | 고정 조건 | 비교 조건 | 다음 단계 |
|---|---|---|---|
| A: degradation | C3, cosine, 동일 checkpoint/split | r=0.75 / 0.50 / 0.25 | degradation 선택 |
| B: feature | 선택한 r, cosine, 동일 checkpoint/split | C3 / C4 / C5 | stage 선택 |
| C: metric | 선택한 r와 stage, 동일 checkpoint/split | cosine / L1 / L2 | 최종 비교 |

A→B→C의 수동 선택을 기본 권고안으로 둠. GUI가 사용자의 선택 전 임의로 다음 실험 조건을 결정하지 않도록 구현함. 선택 조건과 이유, 선택 시점, 근거 run ID를 저장함.

선택 판단에는 작은 객체–BG의 차이, 크기와 score의 관계, bin별 분포, 이미지 간 일관성, 복잡한 배경의 높은 반응을 함께 확인함. discrepancy가 가장 크거나 예상 순서에 가장 잘 맞는다는 이유만으로 좋은 설정이라고 단정하지 않음.

단계 간 중복 조합은 재사용함. best/last 비교에는 동일한 r/stage/metric/split을 적용함. 서로 다른 조건을 선택한 결과를 같은 조건의 checkpoint 비교처럼 표시하지 않음.

## 9. 통계와 Figure

### 9.1 필수 저장값

- BG / VT / T / S / M / L의 mean, std, median, sample count.
- 객체별 원본 크기, 입력 크기, class, image ID, box, discrepancy.
- object size–discrepancy scatter 및 Spearman correlation.
- Pearson correlation은 보조 분석으로 저장함.
- 각 bin과 BG의 절대 차이 및 상대 차이. 상대 차이는 BG 평균이 0에 가까우면 NA로 기록함.
- Kruskal–Wallis 및 필요 시 pairwise Mann–Whitney U는 보조 통계로 저장함. 다중 비교 시 보정 여부를 명시함.

객체는 이미지 안에 군집되어 있으므로 모든 객체가 독립 표본이라는 가정은 강함. p-value만으로 주장을 강화하지 않음. **추가 제안:** confidence interval이 필요하면 이미지 단위 cluster bootstrap을 사용함. 본 실험에서 관찰한 경향을 seed·모델 전체로 일반화하지 않음.

### 9.2 필수 시각화

1. bin별 mean discrepancy bar chart + 실제 숫자 table.
2. BG / VT / T / S / M / L box plot. BG와 객체의 sample 단위 차이를 명시함.
3. object size–discrepancy scatter. x축의 size 정의와 로그 축 여부를 표시함.
4. degradation별, stage별, metric별 비교.
5. best/last 또는 확보된 checkpoint별 동일 조건 비교.

Bar chart의 std는 분산을 나타내며 평균의 신뢰구간과 다름. Figure에 무엇을 표시했는지 적음. Box plot은 사분위수·whisker 정의를 고정함. 표시용 subsampling을 하면 seed와 수를 기록하고 전체 통계는 원본 데이터로 계산함.

Heatmap은 같은 metric을 비교할 때 고정 color scale을 기본으로 함. 이미지별 자동 scale을 제공하면 표시용 옵션임을 명시함. PNG와 논문용 PDF/SVG 등 export를 지원함.

## 10. GUI 요구사항

### 10.1 상단 진행 상태

`Checkpoint Validation / Training / Degradation / Feature Stage / Metric`을 탭 또는 카드로 표시함.

- 초록색 + DONE: 완료.
- 빨간색 + RUNNING: 실행 중.
- 회색 + PENDING: 대기.
- SKIPPED, FAILED, WAITING_FOR_SELECTION은 텍스트와 아이콘으로 구분함.

현재 조건, 완료 목록, 남은 조건, 분석 이미지 수/전체 수, elapsed time, ETA를 표시함. 분석 중인 작업을 detector 학습 중이라고 표시하지 않음.

### 10.2 Training monitor

직접 학습하는 경우 epoch / total epochs, total loss, component losses, validation mAP@[.5:.95], mAP50, learning rate, GPU utilization, VRAM, ETA를 실시간 표시함. 기존 checkpoint를 사용하는 경우 관련 값은 비활성화함.

학습 완료 후 checkpoint 검증 및 analysis viewer로 전환함. 수동 선택이 필요한 ablation 경계에서는 결과를 보여주고 대기함.

### 10.3 Image viewer

- 사용자가 지정한 폴더의 이미지 목록, dropdown, 이전/다음 탐색.
- checkpoint / degradation / stage / metric 선택.
- Original, Degraded, Discrepancy Heatmap, Heatmap Overlay, GT Overlay 비교.
- GT에는 size bin과 실제 객체 score 표시: 예 `VT | D=0.382`.
- 하단에 현재 이미지의 bin별 bar chart와 수치 table.
- best/last가 있으면 둘 다 비교. 하나만 있으면 실제 checkpoint 이름 하나만 표시.
- GT 없는 사용자 이미지도 heatmap은 지원하되 객체 크기별 통계는 unavailable로 표시함.

### 10.4 Dataset-level viewer

현재 split 이름과 선택/최종 평가 용도를 항상 표시함. 전체 split에 대해 ① size별 통계, ② 크기–discrepancy 관계, ③ degradation 비교, ④ stage 비교, ⑤ metric 비교를 지원함. Box plot과 Statistics 화면, CSV/figure 다운로드를 제공함.

### 10.5 실행 구조 — 추가 제안

Streamlit은 표시 및 제어를 담당하고, 학습·분석은 별도 worker process에서 수행함. GUI rerun이나 새로고침이 학습 재시작 또는 중복 job 생성을 유발하지 않도록 함. job ID, 상태 파일, 로그, 중단/재개 정보를 저장함. 실패한 job은 이유를 표시하며 완료로 처리하지 않음.

## 11. 결과물과 코드 구조 — 제안

```text
project/
  EXPERIMENT_PLAN.md
  compatibility_report.md
  configs/
    detector.py
    analysis.yaml
  src/
    data_audit.py
    checkpoint_validation.py
    train.py
    feature_extractor.py
    degradation.py
    discrepancy.py
    region_pooling.py
    statistics.py
    analysis_runner.py
    job_manager.py
  app/
    streamlit_app.py
  scripts/
    audit_dataset.py
    train_detector.py
    validate_checkpoint.py
    run_analysis.py
  tests/
    test_geometry.py
    test_discrepancy.py
    test_aggregation.py
  runs/<run_id>/
    config.yaml
    manifest.json
    selection.json
    run.log
    objects.csv
    background.csv
    summary.csv
    summary.json
    statistics.json
    figures/
    qualitative/
```

`manifest.json`에는 timestamp, seed, git commit, 환경 버전, dataset split/식별값, checkpoint 경로/hash, 전처리, degradation 연산, module mapping, metric, pooling 및 제외 규칙을 저장함.

`objects.csv`에는 최소한 run_id, image_id, object_id, class_id, original_w/h, original_size, input_w/h, input_size, size_bin, checkpoint_id, ratio, stage, metric, score, valid_sample_count를 저장함.

`summary.csv`에는 region, sample_unit, count, mean, std, median, q1, q3 및 제외 수를 저장함. 빈 bin은 0점으로 채우지 않고 NA와 count=0으로 표시함.

Feature cache는 image/preprocessing/checkpoint/ratio/stage를 식별하는 key를 사용함. 대규모 feature 전체 저장은 선택 기능으로 두고, 기본은 이미지 단위 처리와 raw score 저장으로 디스크 사용을 제한함. L1/L2/cosine은 확보된 동일 feature pair로 계산하여 중복 forward를 줄일 수 있음.

## 12. 실행 순서와 완료 조건

1. 환경·데이터·checkpoint compatibility report 작성.
2. checkpoint 검증 또는 baseline fine-tuning.
3. 소수 이미지로 좌표 변환·padding 제외·metric·집계의 의미 검증.
4. 전체 split의 Stage A 실행 및 결과 저장.
5. 선택한 degradation으로 Stage B 실행.
6. 선택한 stage로 Stage C 실행.
7. checkpoint 비교, dataset 통계, GUI와 figure export 검증.
8. 실제 선택한 설정·한계·실행 명령을 README에 기록.

### 의미 있는 검증 항목

- 같은 입력 두 개의 유효 feature cosine discrepancy가 수치 오차 범위에서 0인지 확인함.
- 동일 vector, 상수 차이, zero norm에 대해 metric 정의와 구현이 일치하는지 확인함.
- resize/padding된 GT가 overlay에서 맞는지, tiny/subpixel box에 빈 ROI 오류가 없는지 확인함.
- 인공 discrepancy map으로 객체 동일 가중 평균과 BG 위치 가중 평균을 구분해 검증함.
- 겹친 GT, ignore region, 빈 bin, GT 없는 이미지, 공개 checkpoint 한 개를 처리함.
- 같은 설정의 재실행이 재현 가능한지 확인하고, 비결정적 연산이 남으면 기록함.
- GUI의 수치가 저장된 raw 결과와 일치하고, 새로고침이 job을 중복 실행하지 않는지 확인함.

완료는 “예상 가설이 성립함”이 아니라 **정의한 실험이 재현 가능하게 실행되고 결과·반례·한계가 보존됨**을 뜻함. VT>T>S>M>L 또는 VT>BG가 성립하지 않아도 결과를 삭제하거나 조건을 숨기지 않음.

## 13. 더 발전시킬 수 있는 아이디어 — 기존 합의와 분리

### 우선순위 1: 설정 선택 데이터와 최종 분석 데이터 분리

test 결과를 보고 degradation/stage/metric을 고르면 최종 결과가 낙관적으로 보일 수 있음. **권고:** val에서 A/B/C 설정을 선택하고 GT가 있는 독립 test에서 고정 설정을 최종 보고함. 모든 ablation을 test에서 탐색했다면 탐색적 분석임을 명시함. test GT가 없으면 val을 이미지 단위로 선택/보고 subset으로 나누거나, 전체 val 분석임을 밝히고 별도 일반화 주장을 제한함.

### 우선순위 2: 원본 크기와 입력 크기를 함께 기록

주 분석 bin은 사용자 결정대로 원본 크기를 유지함. 다만 같은 원본 12px 객체도 이미지 resize 비율에 따라 detector에서 차지하는 pixel 수가 달라짐. input_size와 resize_scale을 함께 저장하고 **보조 scatter/상관 분석**에 사용하면 결과 해석에 도움이 됨. 주 분석 bin을 몰래 바꾸지 않음.

### 우선순위 3: “작은 객체”와 “복잡한 texture”의 혼동 확인

discrepancy가 높은 배경이나 큰 객체 경계도 qualitative 결과에 포함함. 필요 시 local gradient/texture 지표와 score의 관계를 보조 분석함. pure-background 확장 실험을 추가할 필요는 없음. 높은 score가 항상 작은 객체를 뜻한다는 과도한 해석을 줄이는 것이 목적임.

### 우선순위 4: 표현 변화가 검출 어려움과 연결되는지 확인

추가 실험 여력이 있으면 동일 GT에 대한 원본/열화 입력의 detection 상태 변화를 저장함. 둘 다 검출 / 원본만 검출 / 열화만 검출 / 둘 다 미검출별 score를 비교하면, 단순 feature 변화와 실제 검출 취약성의 연결을 탐색할 수 있음. IoU·class matching·confidence threshold를 먼저 고정해야 하며 현재 필수 범위는 아님.

### 우선순위 5: 순차 선택의 상호작용 한계 확인

C3에서 선택한 degradation이 C5에서도 최적이라는 보장은 없음. 최종 설정을 고른 뒤 여력이 있으면 degradation×stage의 3×3만 보조 확인함. metric까지 모두 확대하는 grid search는 필수가 아님. 순차 실험으로 얻은 결과를 전역 최적이라고 부르지 않음.

## 14. 구현 전에 남은 결정

| 결정 | 권고 기본안 | 결정 시점 |
|---|---|---|
| 실제 dataset/root 및 이미지 viewer 폴더 | 실행 서버의 경로 검사 | 착수 시 |
| 공개 weight 사용 가능 여부 | config/code/학습 split 검증 후 결정 | 사전 조사 |
| 640 resize/padding 방식 | baseline 확인 후 명시적으로 고정 | 코드 검사 |
| degradation 연산 | 유효 이미지 영역, bilinear + downsample antialias | 전처리 확정 시 |
| size bin 경계와 underflow | 공식 evaluator 대조, underflow 별도 보존 | 구현 전 |
| tiny box pooling | input 좌표계 scalar map + fractional-area 평균 | 구현 전 |
| zero norm | invalid mask와 비율 기록 | metric 구현 전 |
| BG box plot sample | 이미지별 BG 평균, 객체 분포와 단위 차이 명시 | 통계 구현 전 |
| ablation 수동 선택 | GUI 결과 확인 후 다음 조건 확정 | 단계 경계 |
| 선택/최종 분석 split | val 선택, 독립 GT-test 보고 | 분석 시작 전 |

## 15. 후속 구현자에게 전달할 지시

이 문서를 기준으로 코드를 구현하되, 먼저 실제 데이터와 repository를 검사하여 compatibility report를 작성할 것. 확정된 사용자 결정을 임의 변경하지 말 것. 미확정 항목은 위 기본안을 검토하고 실제 적용값을 config와 report에 명시할 것. 공개 VisDrone checkpoint가 검증되면 학습을 생략하고, 그렇지 않으면 COCO pretrained 기반 60 epoch 학습을 수행할 것. 분석은 동일한 고정 detector에서 수행하고 실험마다 재학습하지 말 것. 결과가 가설을 지지하지 않더라도 그대로 저장할 것. 본 요청의 산출물은 계획 MD이며, 이 문서 작성 자체가 원격 서버 실행이나 장시간 학습을 이미 수행했다는 의미는 아님.
