# VisDrone 공개 checkpoint 재조사

확인일: 2026-10-07 (Asia/Seoul). 프레임워크 제한을 풀고 실제 배포 파일과 대응 코드를 확인했다.

## 결론

공개 VisDrone Faster R-CNN checkpoint **두 개를 실제 다운로드했다**.
특히 R50-FPN 후보는 현재 MMDetection detector와 모든 state-dict key/shape가 맞고,
RTX 5090에서 C3/C4/C5 feature 추출 및 원본/열화 비교까지 성공했다.
단, 전체 split mAP 재현이나 학습 provenance의 모든 불일치 해소를 완료한 상태는 아니다.
기존 실험 config와 GUI 기본 checkpoint는 변경하지 않았다.

| 후보 | 확인한 모델 | 파일 확보 | 현재 확인 수준 |
|---|---|---|---|
| VLM4RSDet 공식 배포 | Faster R-CNN / ResNet-50 / FPN / 10 foreground classes | 332,458,705 bytes | 348 state tensors strict 일치, GPU C3/C4/C5 실행 성공 |
| Drone_FasterRCNN | Faster R-CNN / ResNeXt-101-32×8d / FPN (공식 demo config 기준) | 838,370,838 bytes | 다운로드/hash 및 대응 config 확인. legacy serialization 로딩/변환과 평가 미완료 |

## 1. R50-FPN: VLM4RSDet

- 코드: https://github.com/cszzshi/VLM4RSDet
- 코드 commit: `170e51a6cea7d96dbee2fb584ee73447571928d8`
- 공개 weight: https://huggingface.co/cszzshi/VLM4RSDet/tree/main
- HF revision: `ab82169e3514fb114392315a277ed9726380a6f8`
- 파일명: `Faster R-CNN w VLM4RSDet on VisDrone epoch_12.pth`
- 대응 config: https://github.com/cszzshi/VLM4RSDet/blob/main/configs/VLM4RSDet/visdrone/faster-rcnn_r50_fpn_1x_visdrone.py
- 저장소 license: Apache-2.0 명시. HF weight 카드에는 별도 license 메타데이터가 없어 weight 자체의 추가 조건은 확인되지 않음.
- 원본 저장: `data/public_checkpoints/vlm4rsdet_faster_rcnn_r50_visdrone_epoch12.pth`
- 원본 SHA-256: `e293cc91ab6aa62b282b8552309f0591e4a04c387c8f675ee7eaa51e6474eac7`

### 실제 구조 검사

- state groups: backbone 318, neck 16, roi_head 8, rpn_head 6 = 348개.
- bbox classifier `[11,1024]`, bbox regressor `[40,1024]`: 10개 foreground와 background에 대응.
- checkpoint dataset_meta.classes가 현재 프로젝트의 VisDrone 10개 클래스 순서와 정확히 일치.
- 현재 `configs/detector.py` 모델에 missing/unexpected keys 및 shape mismatch 모두 0.
- ResNet layer2/3/4의 실제 출력: C3 `[1,512,48,80]`, C4 `[1,1024,24,40]`, C5 `[1,2048,12,20]`.
- 논문/저장소는 VLM 공동 학습을 설명하지만 inference는 backbone→FPN→RPN/ROI 경로이다.
- 배포 state에는 Florence/adapter tensor가 없으며 표준 detector tensor만 포함되어 있다.

검사 증거: `vlm_standard_r50_strict.json`, `vlm_shapes.json`, `vlm_verification.json`.

### 호환 저장본

`data/public_checkpoints/vlm4rsdet_r50_visdrone_detector_only.pth`를 별도로 생성했다.
모든 learned tensor를 그대로 유지하고 optimizer/history metadata를 제거한 재직렬화본이다.
원본 파일을 보존했다. 오래된 NumPy 경로 별칭 때문에 현재 weights-only loader에서 발생하는
메타데이터 로딩 문제를 피하며, 임의 pickle 로딩(`weights_only=False`)은 사용하지 않았다.
호환본 hash는 `vlm_verification.json`에 저장했다.

### 남은 검증과 불일치

이 배포본을 **일반적인 detector-only 학습 baseline이라고 단정할 수 없다**.
공식 명칭은 VLM4RSDet이고, checkpoint 내부 work_dir에는 `...faster-rcnn_r50_1x_visdrone_baseline`이 들어 있다.
파일명/디렉터리 이름만으로 실제 학습 기법을 확정하지 않는다.

현재 공개 config와 checkpoint 내 cfg 사이에 다음 차이가 있다.

| 항목 | 현재 공개 config | checkpoint 내부 cfg |
|---|---|---|
| Resize | `(1520,1002)`, keep_ratio=True | `(1520,1002)`, keep_ratio=False + Pad |
| validation/test 경로 | `val/` | `test/` |
| train 경로 | `train/` | `train/` |

checkpoint에는 epoch=12, iter=9708이 기록되어 있다. 내부의 `test/`가 보유 데이터의 어떤
split과 동일한지는 입증되지 않았으며, test가 validation으로 사용된 정황을 보존한다.
따라서 이를 독립 final-test로 사용했다고 주장해서는 안 된다.
원본 설정 재현 평가와 본 실험의 640 분석 설정은 구분해야 한다.
현재 full-split mAP 평가 및 논문 수치 재현은 미실행이다.

## 2. maskrcnn-benchmark: Drone_FasterRCNN

- 코드: https://github.com/oulutan/Drone_FasterRCNN
- 코드 commit: `70b69804a7e6f719414f7018e55b156a15d1a62a`
- weight: https://drive.google.com/file/d/1SCJf2JJmyCbxpDuy4njFaDw7xPqurpaQ/view
- 대응 config: https://github.com/oulutan/Drone_FasterRCNN/blob/master/drone_demo/e2e_faster_rcnn_X_101_32x8d_FPN_1x_visdrone.yaml
- 저장소 license: MIT. weight의 별도 license 표기는 확인되지 않음.
- 원본 저장: `data/public_checkpoints/visdrone_model_0360000.pth`
- SHA-256: `096ed875726c77d8acc01e36813ad865ce0e76ddc1b2aa3126625cca8e1b91d3`

Google Drive의 공개 대용량 다운로드 확인 페이지를 거쳐 실제 바이너리 다운로드에 성공했다.
라이브러리 이름은 maskrcnn-benchmark이지만 demo는 `MASK_ON=False`인 Faster R-CNN이다.
config의 depth=101, NUM_GROUPS=32, WIDTH_PER_GROUP=8은 ResNeXt-101-32×8d를 의미한다.
훈련 데이터 이름은 `visdrone_det_cocostyle_augmented_train`, 평가 이름은
`visdrone_det_cocostyle_val`이다. 정확한 augmentation 원본과 split 구성은 미확인이다.

기본 class head는 81이고 predictor가 VisDrone 이름과 COCO placeholder를 섞어 사용한다.
이는 소스 config 기준이며 checkpoint tensor shape로 검증된 값이 아니다.
현재 PyTorch weights-only 로더는 이 legacy serialization을 unsupported operand 73으로 거부한다.
실제 state-dict 대응/클래스 mapping 및 GPU 실행 검증은 아직 수행하지 않았다.
이 weight를 쓰려면 R50 고정 조건을 X101로 변경하고 원래 모델 의미를 보존하는 adapter 또는 이식이 필요하다.

## 다른 조사 후보

- https://huggingface.co/huynoob1312/detectron2-cascade-rcnn-visdrone : `model_final.pth` 및 training metrics 파일 존재. 대응 config/model card가 공개 파일 목록에 없어 우선순위를 낮춤. Cascade R-CNN은 요청한 기본 Faster/Mask R-CNN과도 다름.
- https://github.com/Hiyuur/SR-TOD : 확인한 README/release에서 standard baseline weight 배포를 확보하지 못함.
- https://github.com/amiman/VisDrone : 확인한 repository tree/release에 weight 파일 없음. 학습 코드 제공을 checkpoint 배포로 간주하지 않음.
- https://github.com/Ahb1104/Visdrone_Small_Object_Detection : 확인한 tree/release에 weight 파일 없음.
- https://github.com/AbitGo/TrUoD : R50 VisDrone 링크는 DETR 계열 연구 모델이므로 Faster R-CNN weight로 분류하지 않음.
- 독립 Mask R-CNN VisDrone weight+config 세트는 이번 조사에서 확보하지 못함. COCO pretrained 링크를 VisDrone-trained weight로 혼동하지 않음.

현재 조사 범위에서 **R50 구조 유지에는 VLM4RSDet 배포본이 가장 구체적인 후보**다.
다만 standard training baseline 여부와 평가 split/전처리 불일치 때문에,
지금의 성공 상태는 '파일 확보 + 구조 검사 + feature GPU smoke 통과'이며 '실험용 최종 검증 완료'는 아니다.
