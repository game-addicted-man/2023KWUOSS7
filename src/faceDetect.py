from deepface import DeepFace
import matplotlib.pyplot as plt
import pytesseract
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import os
from glob import glob

# Opencv 사용에 따른 Apache License 2.0 게시.
"""
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

# deepface 사용에 따른 MIT LICENSE 게시.
"""
MIT License

Copyright (c) 2019 Sefik Ilkin Serengil

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# haarcascade_frontalface_default.xml 얼굴인식 데이터셋 파일 사용에 따른 Intel License Agreement 게시.
"""
  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.

  By downloading, copying, installing or using the software you agree to this license.
  If you do not agree to this license, do not download, install,
  copy or use the software.
  
                        Intel License Agreement
                For Open Source Computer Vision Library

 Copyright (C) 2000, Intel Corporation, all rights reserved.
 Third party copyrights are property of their respective owners.

 Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:

   * Redistribution's of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.

   * Redistribution's in binary form must reproduce the above copyright notice,
     this list of conditions and the following disclaimer in the documentation
     and/or other materials provided with the distribution.

   * The name of Intel Corporation may not be used to endorse or promote products
     derived from this software without specific prior written permission.

 This software is provided by the copyright holders and contributors "as is" and
 any express or implied warranties, including, but not limited to, the implied
 warranties of merchantability and fitness for a particular purpose are disclaimed.
 In no event shall the Intel Corporation or contributors be liable for any direct,
 indirect, incidental, special, exemplary, or consequential damages
 (including, but not limited to, procurement of substitute goods or services;
 loss of use, data, or profits; or business interruption) however caused
 and on any theory of liability, whether in contract, strict liability,
 or tort (including negligence or otherwise) arising in any way out of
 the use of this software, even if advised of the possibility of such damage.
"""

class faceDetect:
    def __init__(self, db):
        self.models = ["VGG-Face",
                       "Facenet",
                       "Facenet512",
                       "OpenFace",
                       "DeepFace",
                       "DeepID",
                       "ArcFace",
                       "Dlib",
                       "SFace", ]

        try:
            if not os.path.exists(db):
                os.makedirs(db)
        except OSError:
            print("디렉토리 생성 오류. "+db)


        self.database = db

    def match_face(self, img1, img2=[], name=None):
        # precondition : img1에는 현재 사진 데이터(or 경로), img2에는 민증사진 데이터(or 경로), name에는 이름을 할당
        #                민증 사진을 주기 전에 생년월일로 성인인지 판별 후 성인일때만 함수 이용해야 한다.
        # postcondition: 민증 사진 없이 현재 사진만 입력하면 데이터베이스에 있는 사진들과 비교,
        #                두개 사진 모두 입력하면 2개의 사진 비교 일치하면 데이터베이스에 저장후 true 반환.
        try:
            # img1 이미지 하나만 할당 했을 경우
            if isinstance(img1, str):
                # 경로 입력시 해당 경로 이미지 데이터 가져오기
                img1 = cv2.imread(img1)

            if len(img2)==0:
                # 사진 하나만 인자로 함수를 실행했을때
                # DeepFace.find() 함수 실행시 생기는 .pkl 파일 삭제
                mustBeRemovedFile = self.database+"*.pkl"
                for f in glob(mustBeRemovedFile):
                    os.remove(f)

                # 데이터베이스에서 입력 받은 사진과 일치하는 얼굴이 있는지 판독하기
                dfs=pd.DataFrame()
                dfs = DeepFace.find(img_path=img1,
                                    db_path=self.database, model_name=self.models[2])[0]
                # 일치하는 얼굴이 있을시 데이터베이스에 어떤 사진 데이터가 일치하는지 출력 후 true or false 반환
                print(dfs)
                if dfs.empty:
                    return False

                return True

            # 사진 데이터 2개를 인자로 받아 함수를 실행 시켰을 때
            if isinstance(img2, str):
                # 경로 입력시 해당 경로 이미지 데이터 가져오기
                img2 = cv2.imread(img2)

            # 입력 받은 2개의 사진에서 얼굴 비교하기
            result = DeepFace.verify(img1_path=img1, img2_path=img2, model_name=self.models[6])

            if result["verified"]:
                # 얼굴이 일치하면 데이터베이스에 저장
                cv2.imwrite(self.database + "/" + name + "_pic" + ".jpg", img1)
                cv2.imwrite(self.database + "/" + name + "_ifm" + ".jpg", img2)
                return True

            return False

        except ValueError as v:
            # 얼굴 인식이 되지 않을때 예외 처리
            print(v)
            return False

    def show_face(self, img):
        # 이미지의 있는 얼굴들 출력
        try:
            faces = DeepFace.extract_faces(img, target_size=(224, 224), detector_backend="opencv")
            fig, axs = plt.subplots(int(len(faces) / 2 + 1), 2, figsize=(15, 10))
            axs = axs.flatten()

            for i, face in enumerate(faces):
                IMG = face["face"]
                axs[i].imshow(IMG)

            plt.show()

        except ValueError as v:
            print(v)

    def current_face(self):

        # precondition : "captured.png" 사진 파일이 있어도 되고 없어도 된다. 이 기능의 목적은 사용자의 사진 촬영에 한한다.
        # postcondition: 3초 간의 얼굴 인식 후 촬영되면 "captured.png" 사진 파일이 생성 또는 갱신됨. (현재, 리턴 값은 없음)

        pictureTime = 3  # 사진촬영 대기 시간은 3초.

        camera = cv2.VideoCapture(0)  # 실행하는 장치의 카메라를 켜고 640 X 480 의 해상도로 설정한다.
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        faceRecognize = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        # 원활한 얼굴 인식을 위해 얼굴 인식 데이터 셋이 필요하기 때문에 Intel Corporation 의
        # haarcascade_frontalface_default.xml 파일을 사용하기로 하였다.

        while True:
            isCamera, captured = camera.read()  # 카메라로 captured 에 현재 영상 받아옴.
            grayCamera = cv2.cvtColor(captured, cv2.COLOR_BGR2GRAY)  # 카메라로 받아온 영상을 흑백영상으로 바꿈.

            recognizedFaces = faceRecognize.detectMultiScale(grayCamera, scaleFactor=1.01, minNeighbors=100,
                                                             minSize=(20, 20))
            # scaleFactor 를 1에 가깝게 하면 정확도가 오르지만 처리시간이 길어짐
            # minNeighbors 를 높이면 검출에 있어 좋아지지만 오탐지율도 상승함.

            captured = Image.fromarray(captured)  # 카메라로 촬영되는 영상을 송출할 수 있는 형태로 변환함.
            captured = np.array(captured)

            print(pictureTime)

            if len(recognizedFaces):  # 얼굴이 인식될 때 pictureTime 가 1씩 줄어들고 -1이 되면
                pictureTime -= 1  # Pictured! 문구와 함께 사각형이 빨간색으로 점등하며 사진이 저장된다.

                if pictureTime == -1:
                    cv2.imwrite('captured.png', captured)
                    for x, y, w, h in recognizedFaces:
                        cv2.rectangle(captured, (x, y), (x + w, y + h), (0, 0, 255), 2, cv2.LINE_4)
                    cv2.imshow("original", captured)
                    print("Pictured!")
                    cv2.waitKey(1000)
                    break

            if len(recognizedFaces):  # 얼굴을 인식하면 하얀색 사각형이 얼굴 주변에 위치한다.
                for x, y, w, h in recognizedFaces:
                    cv2.rectangle(captured, (x, y), (x + w, y + h), (255, 255, 255), 2, cv2.LINE_4)

            cv2.imshow("original", captured)  # 영상과 사각형이 합쳐진 형태가 송출된다.

            if cv2.waitKey(1000) == ord('q'):  # 1초마다 작업을 반복하고, 'q' 가 눌리는 경우 프로그램이 종료된다.
                break

        camera.release()
        cv2.destroyAllWindows()
        return captured

