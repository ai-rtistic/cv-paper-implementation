# FPN
## Feature Pyramid Networks for Object Detection

![image](https://user-images.githubusercontent.com/84179578/162136936-dddb5667-2c74-43e7-8c85-3db276f7e7fd.png)


---

기본 용어  

![image](https://user-images.githubusercontent.com/84179578/162136522-e5dc4d73-1ec5-4bc3-b7f9-65639b17bc41.png)

- High resolution image - Low level feature  

→ input image 와 크기가 비슷할 수록  **High resoulution** 이라고 하고, 해당 feature map (layer) 에서는 작은 receptive field 를 가지므로 세밀한**(fine**) 특징들을 찾아내는 역할을 함 → **low-level feature**  

- Low resolution image - High-level feature  

→ input image 에서 conv, downsampling  layer 를 거치며 feature map 의 크기(h,w)가 작아질 수록 **Low resolution** 이라고 하고, 해당 feature map (layer) 에서는 큰 receptive field 를 가지므로 전반적인(**coarse**) 특징들을 찾아내는 역할을 함 → **high-level feature**  

<br/>

- Scale  

![image](https://user-images.githubusercontent.com/84179578/162136783-56ff30f2-1050-44f4-b1b2-a5443ac57554.png)


→ 다양한 크기의 객체를 탐지하기 위해 다양한 scale 를 이용해야함  

→ 이때 window scale 과 image size 는 반비례함 (ex. window size 가 고정일때, image  size 가 커지면 window가 보는 scale 은 작아짐  

<br/>


- **scale invariance**  

→  이미지의 크기(scale) 이 변하더라도  변하지 않는 object feature  

<br/>


**기존 방식**  

![image](https://user-images.githubusercontent.com/84179578/162136814-c9f415ad-d7c4-4929-903c-0af42cc9291a.png)


<br/>


- **Featurized image  pyramid**  

→ 다양한 scale 의 input 이미지를 각각 넣고 예측함  

→ 단점: 속도가 느리고 자원이 많이 필요함  

→ ex. OverFeat  

<br/>


- **Single feature map**  

→ 하나의 모델을 통해 한번에 예측  

→ 속도 빠름  

→ 단점: 작은 size의 객체를 잘 인식하지 못함  

→ ex. YOLOv1

<br/>


- **Pyramidal feature hierarchy**

→ 각 레이어에서 다양한 size 를 가지는 feature map 을 추출해 예측  

→ 단점: 각 레이어에서 나온 feature map 들 간의 **semantic gap** (의미론적 차이)  존재 → low-level feature, high-level feature 간의 차이라고 생각하면됨  

<br/>


### ****Feature Pyramid Network****

![image](https://user-images.githubusercontent.com/84179578/162136870-46619fbc-4511-4765-b414-25d71ccd5e7f.png)
![image](https://user-images.githubusercontent.com/84179578/162136891-f6e63d20-9cdb-495a-a23c-251d02fb5950.png)


single image  를 input 으로 넣어 각 중간 레이어에서 다양한 size 의 feature map 을 추출하고 (**Bottom-up pathway)** upsampling 을 하며 (**Top-down Pathway**) 각 중간 레이어에서 추출한 feature map 을 합쳐주는(**Lateral connections)** 구조  

<br/>


![image](https://user-images.githubusercontent.com/84179578/162136978-9c6b0fc2-0a29-48ef-b6dd-f2bfcc71963b.png)



이때, 각 레이어에서 나온 feature map 의 channel 수를 1x1 conv 를 이용해 모두 동일하게 256x256  으로 맞춰주는 과정을 가짐  

또한, upsampling 시에는 **nearest neighbor** 방법을 사용해 upsampling 함  

FPN 방법을 RPN, Faster RCNN 에 적용시켜 SOTA 성능을 보임  