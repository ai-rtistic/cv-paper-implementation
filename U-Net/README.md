# U-Net

Olaf Ronneberger, et al. "__U-Net: Convolutional Networks for Biomedical Image Segmentation__"  arXiv preprint arXiv:1505.04597 (2015). [[pdf]](https://arxiv.org/pdf/1505.04597.pdf)  


### KET POINT

1. **fully connected layer 없이 conv layer 로만 구성된 FCN 기반의 U-Net**

![image](https://user-images.githubusercontent.com/84179578/162614529-8b5fee63-bc43-4825-9ba9-d1601e2bfd8a.png)


→ Conv layer 로만 네트워크를 구성해 input image 의 공간적인 정보 최대한 유지

→ downsampling + upsampling 의 대칭적인 구조 + 각 층에서 나온 다양한 scale 의 feature map 을 연결해줌(여러 단계의 feature map 정보를 이용)

→ image 의 fine 한 특징과 coarse 한 특징을 모두 사용함→ 전체 context 를 보며 세세한 localization 가능

![image](https://user-images.githubusercontent.com/84179578/162614540-ea0c3ce5-994f-43c2-b29f-4f6a46aeaf29.png)
![image](https://user-images.githubusercontent.com/84179578/162614547-31b96292-abb4-41e5-9f26-5fe02964404c.png)
![image](https://user-images.githubusercontent.com/84179578/162614556-ac74596e-d714-4117-b5f0-856297066446.png)



이때, 최종 ouput 인 Segmentation map의 크기는 Input Image 크기보다 작음.  

→ Convolution 연산에서 패딩을 사용하지 않았기 때문

<br/>


1. **Patches & overlap tile strategy**

기존의 sliding window 방법은 전체 이미지를 특정 window 크기만큼 잘라서 sliding 하며 각각을 CNN 에 통과시킴

→ 중복되는 부분이 너무 많아 연산량이 너무 많음

U-Net 에서는 겹치는 부분 없이 전체 이미지를 일정 개수의 patch 로 나눠 CNN 에 통과시킴

![image](https://user-images.githubusercontent.com/84179578/162614561-6da0eb94-1523-4b61-9708-9b597c15f6e1.png)


이때, 위에서 언급했든이 최종 ouput 인 Segmentation map의 크기는 Input Image 크기보다 작으므로 해당 patch(window) 와 일치하는 segmentation map (ouput) 을 얻기 위해서는 해당 patch 보다 큰 사이즈의 input 을 얻어줘야함

![image](https://user-images.githubusercontent.com/84179578/162614566-0a5076b7-3980-4a2a-b033-27ec0a4f61a2.png)


따라서 input으로 해당 patch (노란색 박스)에 대한 segmentation map(output) 을 얻기 위해 조금 더 큰 사이즈의 input (파란색 박스)을  넣어줌

→ 이때 매 patch(window) 마다 들어가는 input 이 일정 부분이 겹치기 때문에 overlap tile strategy 라고 명명함


<br/>


이때, 전체 이미지에 벗어난 빈 부분은 mirroring 방법으로 채워서 input 이미지로 사용함

![image](https://user-images.githubusercontent.com/84179578/162614569-579e0803-bdd5-4ce0-aaa9-49557305cb39.png)



<br/>



2. **Data Augmentation**

일반적으로 많이 사용하는 data augmentation 기법인 crop, rotation, shift 등은 선형 변환임

→ 실제 세포는 엄청 다양한 형태를 하고 있음 → U-Net 에서는 임의의 비선형 변환 (Elastic Deformation)을 수행함

![image](https://user-images.githubusercontent.com/84179578/162614573-a20124ba-9a1c-4af2-b00f-3b809206ce1d.png)



<br/>



3. **Weight map**

매 ground truth segmentation 마다 weight map 을 계산해 붙어있는 셀 사이의 좁은 공간일 수록 더 높은 가중치를 줌

→ 제일 오른쪽 그림

![image](https://user-images.githubusercontent.com/84179578/162614576-9e7dd843-1f7b-43b6-a458-e4d2d686ce79.png)