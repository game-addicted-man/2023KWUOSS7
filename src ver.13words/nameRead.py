import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract
import re


class CheckCard:
    def __init__(self):
        self.birth = 24
        self.name = ''

    def checkNum(self, img, path):
        pytesseract.pytesseract.tesseract_cmd = path

        plt.figure(figsize=(12, 10))
        img_crop = img[50:450, 100:700]
        # 노이즈 줄이기위해 블러 사용
        img2 = cv2.GaussianBlur(img_crop, ksize=(5, 5), sigmaX=0)

        # 임의의값 아래는 처리 안함,높으면 255로 고정
        img3 = cv2.adaptiveThreshold(img2, 255.0, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)
        result = img3[150:200, 50:300]
        text = pytesseract.image_to_string(result)
        num = re.sub(r'[^0-9]', '', text)
        return num

    def checkName(self, img, path):
        pytesseract.pytesseract.tesseract_cmd = path

        plt.figure(figsize=(12, 10))
        img_crop = img[50:450, 100:700]
        result = img_crop[100:170, 50:170]
        text = pytesseract.image_to_string(result, lang='kor')
        name = re.sub(r'[^가-힣]', '', text)
        return name

    def setBirth(self, path, cap):
        bool_Birth = False
        bool_Name = False
        birth_List = []
        name_List = []
        while True:
            ret, frame = cap.read()
            # 가로,세로
            cv2.rectangle(frame, (100, 50), (700, 450), (0, 0, 255), 3)
            cv2.imshow('frame', frame)

            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))

            # 높이,너비,채널 저장
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if (not bool_Birth):
                agent_num = self.checkNum(img, path)
                if (len(agent_num) == 13):
                    birth_List.append(agent_num[:6])
                    # print(agent_num[:6]) 숫자 확인용
                for i in birth_List:
                    if (birth_List.count(i) >= 3):
                        self.birth = i
                        bool_Birth = True
                        break

            if (not bool_Name):
                agent_name = self.checkName(img, path)
                if (len(agent_name) == 3):
                    name_List.append(agent_name)
                for i in name_List:
                    if (name_List.count(i) >= 3):
                        self.name = i
                        bool_Name = True
                        break

            if cv2.waitKey(1) == ord('q'):
                return
            if (bool_Name and bool_Birth):
                return

    def getBirth(self):
        return self.birth

    def getName(self):
        return self.name

    def takePicture(self, cap, savePath):
        ret, image = cap.read()
        cropped_img = image[90:310, 420:650]
        improve1 = cv2.detailEnhance(cropped_img, sigma_s=10, sigma_r=0.15)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        improve3 = cv2.filter2D(cropped_img, -1, kernel)
        return cropped_img

    def run(self, path):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)  # 가로
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)  # 세로
        self.setBirth(path, cap)
        img = self.takePicture(cap, "./img/")
        cap.release()
        cv2.destroyAllWindows()
        return img