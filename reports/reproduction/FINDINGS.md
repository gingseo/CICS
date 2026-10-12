# Faster R-CNN R50-FPN 성능 격차 조사 (2026-10-08)

**실행 정책 변경:** 이후 사용자 요청으로 최종 선택 기준을 val에서 test AP로 변경했다. 아래 val 선택 계획은 변경 전 기록이다. 현재는 보존된 4 epoch 이후 checkpoint를 test로 비교하고 같은 test에서 보고한다. 1–3 epoch은 기존 보관 정책으로 삭제되어 비교 대상이 아니다. 최종 결과는 test-selected이며 독립 holdout 성능으로 표시하지 않는다. 실제 상태는 `runs/reproduction_r50_ms20/pipeline.json`을 따른다.

## 실제로 확인한 원인

기존 run은 긴 변 640, 비율 유지 입력이다. 16:9 영상은 약 640×360이 된다.
HMF-DEIM Table 3의 Faster-RCNN-R50-FPN 행은 입력 (768, 1344), AP 19.4, AP50 32.9, APs 9.5를 명시한다.
MMDetection의 (1333, 800) 비율 유지 Resize는 대표 val 영상 1360×765를 1333×750으로 만들고,
32 배수 padding 후 1344×768이 된다. 따라서 두 입력은 같은 조건이 아니다.
이전 구현에서 분석용 640 설정을 학습에도 적용한 것이 중요한 차이였다.

같은 checkpoint `runs/training/best_coco_bbox_mAP_epoch_9.pth`, 같은 val 548장,
같은 annotation/class mapping/score threshold/NMS/evaluator에서 해상도만 바꾼 실측:

| 입력 설정 | AP | AP50 | AP75 | APs | APm | APl |
|---|---:|---:|---:|---:|---:|---:|
| 긴 변 640 (기존 epoch 9 로그) | 12.0 | 20.3 | 12.6 | 3.0 | 22.5 | 39.4 |
| 긴 변 최대 1333, 짧은 변 800 | 15.7 | 27.6 | 15.6 | 8.9 | 23.4 | 27.7 |

단위 %. 두 번째 행은 `eval_best640_at1333.json`의 전체 val 평가 결과다.
이는 해상도 영향의 직접 증거다. 기존 모델이 저해상도로 학습됐기 때문에 고해상도 추론만으로
대형 객체 성능까지 개선되지는 않았다. 학습 해상도도 함께 바꾸는 실험이 필요하다.
실제 입력 크기와 epoch별 loss가 감소하는 현상만으로 전체 격차의 원인을 단정하지 않는다.

## 논문 및 GitHub 확인

| 논문 | 확인 내용 | 공개 Faster R-CNN 재현 가능성 |
|---|---|---|
| PFE-Det, Table 2 | test-dev 1610장, AP 19.4/AP50 32.9. 본문 §4.4에서 비교 모델은 각자의 official default 설정으로 학습하며 비균일 reference comparison이라고 설명 | 본문에서 자체 모델/Faster R-CNN 코드·가중치 링크를 확인하지 못함. 데이터 링크만 확인 |
| HMF-DEIM, Table 3 | test 1610장, Faster R-CNN **768×1344**, AP 19.4/AP50 32.9 | 접근 가능한 원문에서 Faster R-CNN config/가중치 링크를 확인하지 못함 |
| Vectorized IOU, Table 4 | R50-FPN AP 19.7/AP50 35.8. 본문 §4는 train/val을 명시 | 저자 저장소는 YOLOv5 계열. 재귀 파일 목록에서 Faster R-CNN 파일/체크포인트 없음, releases 없음 |
| MoE-RelationNet++, Table 8 | R50-FPN AP 21.4/AP50 40.7. 20 epoch, LR 감소 16/19, 기본 평가 Resize 1333×800 | 저자 저장소에 ATSS/FCOS/RetinaNet 설정은 있지만 Faster R-CNN 설정/가중치 없음. VisDrone ATSS config는 다중 스케일 Resize [(1333,640),(1333,1280)] 사용 |
| USF-DETR, Table 5 | R50 AP 20.5/AP50 34.2/AP75 21.9, APs 10.0/APm 29.5/APl 43.3. 본문의 640×640은 제안 모델의 설정이며 Faster R-CNN의 전처리를 특정할 수 없음 | 본문에서 Faster R-CNN config/체크포인트 링크를 확인하지 못함. 표의 평가 split도 확정하지 않음 |

AP/AP50가 두 논문에서 같다는 사실만으로 독립 재현이나 같은 가중치 사용을 추론하지 않는다.
공개 저장소의 ATSS를 Faster R-CNN이라고 바꾸어 부르거나 VLM 보조학습 모델을 vanilla baseline으로 사용하지 않는다.

출처:
- PFE-Det: https://doi.org/10.3390/s26155003 (§4.1/§4.4, Table 2; 로컬 `sources/pfe.pdf`)
- HMF-DEIM: https://pmc.ncbi.nlm.nih.gov/articles/PMC13075290/ (Table 3; `sources/hmf_readable.txt`)
- VIOU: https://doi.org/10.3390/s23063061 ; https://github.com/jijiehao123/aerial-detection (commit fb285d41952dd3e2e3b9a9caca3d013aa204228c)
- MoE: https://doi.org/10.3390/app16115192 ; https://github.com/hani-yu/MoE-RelationNet- (commit b964ca39bf09b1ebd44528a515127b1b2dc38800)
- MoE VisDrone 설정: https://github.com/hani-yu/MoE-RelationNet-/blob/b964ca39bf09b1ebd44528a515127b1b2dc38800/configs/atss_r50_visdrone_2x.py
- USF-DETR: https://doi.org/10.3390/sym17020242 (Table 5; `sources/usf.pdf`)
- 표준 MMDetection 전처리: https://github.com/open-mmlab/mmdetection/blob/v3.3.0/configs/_base_/datasets/coco_detection.py

## 실행 중인 새 실험

- 설정: `configs/detector_r50_visdrone_ms20.py`
- 모델: standard Faster R-CNN / ResNet-50 / FPN, 10 classes. 추가 VLM/attention/DCN 없음.
- 초기화: COCO detector, 기존 640 fine-tuning 가중치를 이어 학습하지 않음.
- train: 6471장, 다중 스케일 [(1333,640),(1333,1280)], keep_ratio, flip 0.5.
- val: 548장, (1333,800), keep_ratio, 동일한 COCO-style evaluator.
- FP32, batch 2, seed 128, SGD LR .0025, momentum .9, decay .0001, grad clip 35.
- 20 epochs; 500 iteration warmup; LR ×.1 at 16/19.
- val 최고 AP로 checkpoint 선택, 선택한 하나의 checkpoint로 최종 test 1610장 평가.
- `runs/reproduction_r50_ms20/pipeline.json`에서 실행 상태와 결과 확인.
- 학습 완료 후 `curves/`, `eval_val/report.json`, `eval_test/report.json` 자동 생성.

이는 공개 recipe를 참조한 표준 Faster R-CNN 개선 실험이다. MoE 논문은 ImageNet 초기화,
다중 GPU LR .01 및 다른 detector를 공개했으므로 동일 코드·동일 학습 조건의 정확 재현이라고 주장하지 않는다.
기존 60 epoch run 및 분석용 640 설정은 보존했다. 아직 19.4–21.4 AP 달성을 확인한 상태가 아니다.

## 비교의 한계와 기록

- 논문의 test/test-dev 수치를 현재 val AP와 동일 조건이라고 표시하지 않는다.
- 로컬 원본은 YOLO export다. 원래 VisDrone ignored regions/others 정보가 없어 공식 ignore 처리를 복원할 수 없다.
- 현재 COCO evaluator의 proposal_nums=(100,300,500), 검출 max_per_img=500, score_thr=.05.
  pycocotools stats[0] (전체 AP)는 maxDets=100; AP50/AP75/크기별 AP는 마지막 maxDets=500 설정을 사용한다.
  이는 현재 실험 간에는 동일하지만 모든 논문과 동일한 평가 프로토콜이라고 단정할 수 없다.
- 별도의 공식 VisDrone challenge AP와 혼동하지 않는다.
- 과거 monitor history는 validation epoch +1 오류가 있었다. 실제 MMEngine 로그는 정확하며,
  이번 학습부터 hook 기록을 고쳤다. 기존 로그/수치를 덮어쓰지 않았다.

검증: 실제 train/val/test pipeline 로딩 및 입력 shape 확인, 6471/548/1610장 확인, validation epoch hook 점검 통과. checkpoint/job 회귀 검사 4개 통과.

초기 실행 확인: 새 실험 첫 epoch 전체 val AP=16.2%, AP50=28.9%, APs=9.3%; monitor history epoch=1로 정상 기록. 현재 학습 진행 중이며 최종 결과가 아니다.
