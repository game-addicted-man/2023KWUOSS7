from faceDetect import faceDetect
from nameRead import CheckCard

def example():
    hi = CheckCard()  # 민증에서 데이터 추출하는 클래스

    # 현재 얼굴인식 및 얼굴 비교하는 클래스
    # 생성자로 데이터베이스의 경로를 입력받는다.
    fd = faceDetect("./image")

    print("민증이 있으시나요?[1/0] : ", end="")
    choice = int(input())

    now_img = fd.current_face()  # 현재 얼굴 데이터 추출
    if choice:

        # 민증에서 얼굴 추출
        # tesseract.exe파일 경로를 인자로 전달해야함.
        card_img = hi.run("/opt/homebrew/Cellar/tesseract/5.3.3/bin/tesseract")

        # faceDetect으로 인식한 얼굴 출력
        fd.show_face(now_img)


        # match_face() 호출시 인자가 3개이면 입력 받은 2개의 얼굴 데이터를 비교 후 같은 인물이면 데이터베이스에 저장 후 True 반환
        result=fd.match_face(now_img, card_img, hi.name)
        print(result)
    else:

        print("데이터베이스에 해당 인물이 있는지 확인중...")
        print("결과 : ", end="")

        # match_face() 호출시 인자가 하나이면 데이터베이스에 저장되어있는 얼굴들을 비교하며
        # 일치하는 얼굴이 있을시 True반환
        result=fd.match_face(now_img)
        print(result)




if __name__ == '__main__':
    example()

