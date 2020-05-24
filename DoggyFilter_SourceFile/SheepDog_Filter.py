import cv2
eye_detector=cv2.CascadeClassifier('data/haarcascade_eye_tree_eyeglasses.xml')
face_detector = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
cap=cv2.VideoCapture(0)
while(cap.isOpened()):
    _,img=cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_detector.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        ex=x-10
        ey=y-50
        ew=w+50
        eh=h+10
        mask = cv2.imread('Image/sheepdog.png',cv2.IMREAD_UNCHANGED)
        mask=cv2.resize(mask,(ew,eh))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        w,h,c=mask.shape
        for i in range(0,w):
            for j in range(0,h):
                if mask[i, j][3] != 0:
                    img[ey + i,ex + j] = mask[i, j]
    cv2.imshow("SheepDog Filter",img)
    k=cv2.waitKey(10)
    if k == ord('q'):
        break
    if k == ord('s'):
        cv2.imwrite("SheepDog_Filter.png",img)
cap.release()
cv2.destroyAllWindows()
