import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract
import re
import faceDetect
import nameRead
import os
from glob import glob

if __name__ == '__main__':

    database_path = "./database"

    if not os.path.exists(database_path):
        os.mkdir(database_path)

    hi = nameRead.nameRead()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800) # 가로
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800) # 세로
    hi.getBirth(r"C:\Program Files\Tesseract-OCR\tesseract.exe",cap)
    cap.release()
    cv2.destroyAllWindows()

    #이름 3번 인식 시 카메라로 사진을 찍어서 주민등록증의 사진 추출 후 face.jpg 파일로 저장
    image = cv2.imread("./database/card.png")
    cropped_img = image[90:310, 420:600]
    cv2.imwrite("./database/face.png",cropped_img)
    #정보의 양을 늘리는 것(즉, 해상도를 늘리는 것)은 불가능하기에 필터링으로 이미지를 선명하게 변환
    improve1 = cv2.detailEnhance(cropped_img, sigma_s=10, sigma_r=0.15)
    cv2.imwrite("./database/improved_face.png", improve1)

    #sharpening(선명하게)한 것, 제일 쓸만한듯
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) 
    improve3 = cv2.filter2D(cropped_img, -1, kernel) 
    cv2.imwrite('./database/sharpened_face.png', improve3)

    mustBeRemovedFile = './database/*.pkl'  # image 폴더 내의 *.pkl 확장자 파일을 삭제해주어야 함.

    for f in glob(mustBeRemovedFile):
        os.remove(f)

    fd = faceDetect.faceDetect("./database")
    # # fd.face_show("/Users/choejm/image/jaemin0.jpg")
    # print(fd.match_face(img1="image/MESSI.jpg"))
    # print(fd.match_face("image/jonhun.jpg", "image/jaemin0.jpg", name="일론머스크"))
    
    now_img = fd.current_face()
    print(fd.match_face(now_img))
    print(fd.match_face("./database/sharpened_face.png", "captured.png", name=hi.name))
    print(fd.match_face("./database/improved_face.png", "captured.png", name=hi.name))
    print(fd.match_face("./database/face.png", "captured.png", name=hi.name))
