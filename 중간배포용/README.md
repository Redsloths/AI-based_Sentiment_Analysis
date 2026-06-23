1. 모델 설명(구조 등)
    1) 주 모델: arcface기법 기반의 ViT, EfficientNet모델의 앙상블
        - FER 사용했으나, 정확도 문제 있어 arcface 기반으로 변경
        - 기본적인 알고리즘은 Neutral과 4감정(happiness, sadness, panic, anger)의 1차 이진분류 후 4감정의 분류를 진행. (5감정 분류시 Neutral의 이질성으로 인해 점수가 크게 하락)
        - CNN기반 pretrainedmodel: EfficientNet-B2
            * 표정 데이터를 정확하게 읽어오기 위해 crop된 이미지 세세한 분석이 가능한 CNN기반 모델채택
        - ViT모델 
            * Attention 알고리즘이 포함된 모델로, 현재 만능용도로 널리 쓰이는모델이라 채택
        - Model Ensemble 
            * 4감정 분류까지 학습한 EfficientNet-B2모델과 ViT모델의 ensemble작업
            * 두 모델 합성시 Logit 가중치 조절하여 합성 --> 0.5:0:5일때 best
            * Ensemble 이후 혼동행렬 검사 후 anger에 과도하게 판정이 쏠리는 문제 --> Anger의 logit을 -2.1교정

    2) RAW데이터 가공
        - 기본 4감정 분류에서 전체 감정을 5분류로 재분배
            happy, anger, sadness, panic (+ neutral)
        - Bbox데이터 이상한 사진은 자료목록에서 삭제
        - 오분류 된 감정데이터는 수작업으로 재배치(이 과정에서 Neutral로 분류된 사진 약 1000장)
        - 전체 사진 중 표정읽는것에 집중한다고 가정, 원본 데이터에서 '감정분류만' 중점으로 분석 --> 안면인식 BBOX 좌표 기반으로 얼굴사진 crop
        - 여기까지의 데이터로 모델 학습 완료
        - 최종 출력 코드인 emotion_model.py에 TTA코드 첨부 --> 정확도 상승

2. 성능(각 모델별 최고 validation 성능)
    1) EfficientNet: Val Acc: 88.04%
    2) ViT: Val Acc: 0.8563
    3) Ensemble stack: 0.8833

3. 사용 데이터
    1) Pretrained model(ImageNet)
    2) 제공받은 약 7500장 데이터(image, label, npz)
       
4. 실행 방법
    1) vsc 터미널에서 가상환경 생성: . /venv/Scripts/activate (mac: source venv/bin/activate)
    2) 터미널에 'pip install -r requirements.txt' 입력
    3) line9: "test_face.jpg" 부분에 분석할 이미지 입력
    4) RUN