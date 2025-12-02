# 광운대학교 2023-2 오픈소스소프트웨어 7조 프로젝트
#### 2023-2 광운대학교 소프트웨어학부 오픈소스소프트웨어 팀프로젝트 7조 백색 하늘의 혼
조원 : 정승현, 최재민, 배주환, 이종훈

### 프로젝트 소개
-----
무인 주민등록증 검사기

### 계획
-----
![KakaoTalk_20230922_135435240](https://github.com/game-addicted-man/2023KWUOSS7/assets/107955161/6b5c6c12-f346-46f6-ac05-90d049ae7000)

### 소프트웨어 소개
-----
 주민등록증에서 이름과 생일을 추출, 이후카메라를 통해 얼굴을 인식시켜 주민등록증의 얼굴과 실제 얼굴이 동일한지를 출력하는 프로그램
   
**2가지 버전이 존재함**
* 주민등록증의 앞 6자리 (생년월일)만을 읽음 / src 파일
* 주민등록증의 13자리 전부 (주민등록번호)를 읽음 / src ver.13words 파일


### 설치 방법
------

*파이썬 3에서 구동 가능*  
**라이브러리 설치** :
```
pip install numpy
pip install opencv-python
pip install matplotlib
pip install pytesseract
pip install deepface
pip install pillow
```
윈도우 환경에서는 pip install pytessract을 사용하지 않을 시에, tesseract를 따로 exe를 설치 후, 설치 경로를 알아놓아야 한다. 밑은 윈도우 환경에서 사용 시 다운로드 받아야하는 tesseract 파일이다.
```
https://github.com/UB-Mannheim/tesseract/wiki
```  
macOS 환경에서는 터미널에서 설치가 가능하고, 역시 경로를 외워놓아야 한다. 밑의 사용 방법에서 쓰인다.

위 모듈들을 설치한 후, 아래 사진과 같이 저장소의 홈 화면에서 초록색 code 버튼을 누른 후, Download ZIP을 누른다.    




**전부 설치 이후, 파일 실행**   
다운로드 받은 파일들의 압축을 푼 후, src 디렉토리에 example.py 파일을 실행한다.
  
 


### 사용 방법
-----
1. example.py의 코드의 23번째 줄에서 테서렉트 파일의 경로 설정이 필요하다. **card_img = hi.run(“테서렉트 실행 파일 경로”)** 

**※주의※ 윈도우일 경우의 경로설정 시 이런 방식으로 설정해 주어야 함. macOS는 해당없음**  
절대 경로 앞에 r을 입력해 주어야한다.  사진은 example.py 코드의 일부이다. 직접 사용 시 수정이 필요하다.  
<img width="452" alt="보고서사진저장5" src="https://github.com/game-addicted-man/2023KWUOSS7/assets/107955161/6557d837-f7b4-4c5b-af27-43f4db843b00">
<img width="452" alt="보고서사진저장6" src="https://github.com/game-addicted-man/2023KWUOSS7/assets/107955161/8c2c3a9d-7532-4a9d-9e9a-f5e0e3f31978">  
위 사진은 윈도우 환경에서, 아래 사진은 macOS 환경에서 실행경로를 수정한 모습이다.

이때 run() 함수에 테서렉트 exe파일 경로를 명시해줘야 한다.

![보고서사진저장2](https://github.com/game-addicted-man/2023KWUOSS7/assets/107955161/430205da-c049-44f1-b5f9-e728a51e74a6)

2. 1(주민등록증 존재) 혹은 0(주민등록증 미존재)를 입력 시 얼굴 인식 카메라가 생긴다.

3. 얼굴 인식 완료 후 주민등록증 인식용 카메라가 생김, 이 카메라에는 빨간 선이 있는데, 빨간 선에 맞춰서 주민등록증을 가져다 댄다. 그러면 정보가 user에 저장된다.




저장된 정보는 다음 코드로 확인할 수 있다.

user.run()	 : 프로그램을 실행하고 민증 사진을 반환

user.getName()	 : 민증 이름을 반환

user.getBirth()	 : 민증 생년월일(주민번호 앞자리) 반환

user.BirthCheck(): 민증의 정보가 성인이면 true 아니면 false반환 


4. 인식 완료 이후 주민등록증의 얼굴과 인식된 얼굴이 동일하면 true, 동일하지 않으면 false가 출력된다. 
 

### 사용 언어
-----
python3

### 사용 라이브러리
-----
* numpy
* cv2
* matplotlib.pylot
* pillow(PIL)
* tesseract
* re(regular expression)
* deepface

### 사용 라이선스
-----
* BSD License (Matplotlib, NumPy)
* Apache License 2.0 (tesseract, Opencv)
* HPND License (Pillow)
* MIT License (deepface)
* Intel License (haarcascade_frontalface_default.xml 얼굴인식 데이터셋 파일 사용)
