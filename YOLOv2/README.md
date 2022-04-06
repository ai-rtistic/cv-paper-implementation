# YOLO v2

Joseph Redmon, Ali Farhadi. "__YOLO9000: Better, Faster, Stronger__" 	arXiv preprint arXiv:1612.08242 (2016). [[pdf]](https://arxiv.org/pdf/1612.08242.pdf)  



![image](https://user-images.githubusercontent.com/84179578/161790465-7feee6ad-abc8-4cc8-8882-e1b6acfbd4f5.png)


본 논문 구현에서는 논문에서 나온 핵심적인 내용을 그대로 구현했으나 간소화된 부분도 많습니다.  

데이터셋은 [VOC devkit 2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/#devkit) 을 기준으로 코드를 구성했습니다.  

![image](https://user-images.githubusercontent.com/84179578/161883097-fd47fb43-a91d-4b9c-9975-a6e213d536ce.png)



## KEYPOINT : Anchor Box, Jointly Train, 9000 class

<br/>

1.  **Anchor Box 사용**

YOLOv1 에서는 bbox 의 좌표를 fc-layer 를 이용해서 예측했음

→ YOLO v2 에서는 anchor box를 도입해, CNN 을 이용해 anchor box 와 bbox 간의 차이(offset) 을 학습하는 방법을 이용

→ 모델을 더욱 쉽게 학습시킬수 있음

이때, 처음 bbox 를 지정할때, 기존에는 정해진(hand-picked) size 와 종횡비(aspect ratio) 를 가진 achor box 를 사용했음.

→ 처음부터 더 적절한  achor 박스의 크기와 종횡비를 사용하면  offset 을 더 쉽게 학습할 수 있음

따라서 YOLO v2 에서는 train dataset 의 모든 ground truth 에 k-means Clustering 방법을 사용하여 최적의 achor box 의 크기와 종횡비를 결정함. 

이때 k 는 각 grid 마다의 anchor box 의 갯수


<br/>

2. **Hierarchical classification**

기존의 classification task 는 단 하나의 label 을 가질 수 있지만

YOLOv2 는 WordNet 을 이용해 계층적 라벨을 가짐 (ex. 요크셔테리어-개)

<br/>

3. **Jointly Training**

YOLOv2 에서는 classification dataset 과 detection dataset 데이터를 모두 이용해 학습함

detection dataset 은 전체 loss 를 이용해 학습하고

classification dataset 은 중간 까지의 loss (classification loss) 만 사용하여 학습함

<br/>


4. **Passthrough layer** 

YOLOv2 는 13*13 feature map 에서 예측을 수행하는데, 이때, 작은 크기의 물체들을 세심하게 예측하기 위해 finer 한 feature을 사용해야함. 이를 위해 passthrough layer 를 이용해 26*26 size 의 feature 을 연결함(ResNet 같은 개념이라고 생각하면 됨)

*합칠때 reshape 를 해서 concat 함*

<br/>


5. **High Resolution Classifier**

YOLO v1 모델은 Darknet을 224x224 size 의 ImageNet 이미지를 이용해  pre-train시킨 가중치를 불러와서 이후 detection 시  transfer learning 할때에는 448x448 크기의 이미지를 input 으로 사용

→ 부정확함

따라서 YOLO v2 에서는 처음부터 448x448 의ImageNet 이미지로  pretrained 함.

→ mAP 4% 향상

<br/>

6. **Multi-Scale Training**

YOLOv2 는 디양한 size 의 이미지에 robust 해지기 위해 input image size 를 조정하여 다양한 scale의 이미지에서 학습함