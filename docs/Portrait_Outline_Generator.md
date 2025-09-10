# Project Description: Portrait Outline Generator
## 목적 (Goal)

- 초상화 사진을 입력받아, 원본 형태에 충실하면서도 단순하고 따라 그리기 쉬운 아웃라인을 자동 생성한다.
- 사용자가 그림을 시작하기 편하도록 **가이드 라인(세로 중심선 + 눈높이선)**을 함께 제공한다.
- 최종 산출물은 학습/연습용 베이스 라인 아트로 활용되며, 인쇄·디지털 환경 모두 지원한다.

## 핵심 아이디어 (Key Ideas)

1. 형태 보존:생성형 모델 대신 세그멘테이션 + 에지 검출 기반 판별형 접근 사용 → 원본과 동일한 구조 유지

2. 불필요한 디테일 제거:
    - Bilateral / Guided Filtering → Texture 억제
    - Morphology / Small Component Removal → 잡선 제거
    - RDP (폴리라인 단순화) + Chaikin Curve → 따라 그리기 좋은 선으로 정제

3. 가이드 라인 자동 생성
- Mediapipe Face Detection → 양쪽 눈 keypoint 추출 → 눈높이선 생성
- 얼굴 bbox 중심 → 세로 중심선 생성
- 얇은 연필선 느낌의 가이드 표시

## 파이프라인 (Pipeline)
1. 입력 이미지 로드 (OpenCV)
2. 전처리: 색공간 변환 + Edge-friendly smoothing
3. 사람 분리: Mediapipe SelfieSegmentation으로 배경 제거
4. 얼굴 특징 추출: Mediapipe FaceDetection으로 눈 위치/중심선 계산
5. 윤곽선 검출:
    - 기본: OpenCV Canny (자동 임계값)
    - 고급: DexiNed (MIT, optional) **default**
6. 후처리: Morphology Close + 작은 컴포넌트 제거
7. 윤곽선 구조화: Contour → Polyline 단순화(RDP) → Chaikin Curve smoothing
8. 출력:
    - *_construction.png: 세로선 + 눈선 가이드
    - *_outline.png: 단순 윤곽선
    - *_outline.svg: 벡터 형식 (선택)