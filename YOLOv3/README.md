# YOLO v3

-  KEYPOINT : 더 정확하게! (시간은 조금 더 걸리지만 그래도 real time) - FPN 도입, class 별 Logistic classifier 도입

거의 YOLOv2 와 동일

개선점

- Model - DarkNet 53

→ residual block 사용

![image](https://user-images.githubusercontent.com/84179578/163177801-d882e058-9979-449f-9f9f-52b2fb8369d9.png)

- **FPN 사용** → multi scale image 를 학습하기 위해 FPN 적용 → 3가지 다른 scale 의 feature map 을 사용

![image](https://user-images.githubusercontent.com/84179578/163178073-72b228be-53f5-47f1-9937-884b8e037f1e.png)


- softmax  대신 class 별 **logistic classifier** 사용