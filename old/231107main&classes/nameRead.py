import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract
import re

class nameRead:
#img: 추출할 사진 ,path: 테서렉트 실행파일 경로
#주민번호 앞자리 반환
    def __init__(self):
        self.birth=24
        self.name = ''

    def checkNum(self,img,path):
        pytesseract.pytesseract.tesseract_cmd = path

        plt.figure(figsize=(12, 10))
        img_crop=img[50:450,100:700]
        #노이즈 줄이기위해 블러 사용
        img2 = cv2.GaussianBlur(img_crop, ksize=(5, 5), sigmaX=0)

        #임의의값 아래는 처리 안함,높으면 255로 고정
        img3 = cv2.adaptiveThreshold(img2, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19,9)
        result = img3[150:200,50:300]
        text = pytesseract.image_to_string(result)
        num = re.sub(r'[^0-9]','',text)
        return num

    #img: 추출할 사진 ,path: 테서렉트 실행파일 경로
    #주민번호 이름
    def checkName(self,img,path):
        pytesseract.pytesseract.tesseract_cmd = path

        plt.figure(figsize=(12, 10))
        img_crop=img[50:450,100:700]
        result = img_crop[100:170,50:170]
        text = pytesseract.image_to_string(result,lang='kor')
        name = re.sub(r'[^가-힣]','',text)
        return name
        

    #path: 테서렉트 실행파일 경로
    #웹캠 실행후 이름과 생년월일 출력
    def getBirth(self,path,cap):
        bool_Birth = False
        bool_Name = False
        birth_List = []
        name_List = []

        database_path = "./database"

        while True:
            ret, frame = cap.read()
            #가로,세로
            cv2.rectangle(frame,(100,50),(700,450),(0,0,255),3)
            cv2.imshow('frame',frame)

            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))


            #높이,너비,채널 저장
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            #생년월일 추출성공 전까지 수행
            if(not bool_Birth):
                agent_num = self.checkNum(img,path)
                if(len(agent_num)==13):
                    birth_List.append(agent_num[:6])
                    # print(agent_num[:6]) 숫자 확인용
                #같은 날짜를 3번 발견하면 이를 정답으로 간주
                for i in birth_List:
                    if(birth_List.count(i)>=3):
                        print('birth: ', i)
                        bool_Birth = True
                        break

            #이름 추출성공 전까지 수행
            if(not bool_Name):
                agent_name = self.checkName(img,path)
                if(len(agent_name)==3):
                    return_value, image = cap.read()
                    cv2.imwrite(database_path + "card.png", image)
                    name_List.append(agent_name)
                    
                #같은 이름을 3번 발견하면 이를 정답으로 간주
                for i in name_List:
                    if(name_List.count(i)>=3):
                        self.name = i
                        bool_Name = True
                        break
            
            if cv2.waitKey(1) == ord('q'):
                break       
            if(bool_Name and bool_Birth):
                break
