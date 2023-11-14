import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract
import re
from faceDetect import faceDetect
from nameRead import CheckCard

if __name__ == '__main__':
    hi=CheckCard()
    #card_img = hi.run("/opt/homebrew/Cellar/tesseract/5.3.3/bin/tesseract")
    #cv2.imwrite("./card.png", card_img)
    fd = faceDetect("./image")
    # # fd.face_show("/Users/choejm/image/jaemin0.jpg")
    # print(fd.match_face(img1="image/MESSI.jpg"))
    # print(fd.match_face("image/jonhun.jpg", "image/jaemin0.jpg", name="일론머스크"))

    now_img = fd.current_face()
    print(fd.match_face(now_img))
    #print(fd.match_face(now_img, "./card.png", hi.name))
