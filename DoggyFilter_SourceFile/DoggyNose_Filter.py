import cv2
nose_detector=cv2.CascadeClassifier('data/haarcascade_mcs_nose.xml')
face_detector = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
cap=cv2.VideoCapture(0)
while(cap.isOpened()):
    _,img=cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2BGRA)
    nose = nose_detector.detectMultiScale(gray)
    for (nx,ny,nw,nh) in nose:
        nw+=10
        top_left=(nx,ny)
        bottom_right=(nx+nw,ny+nh)
        center=(nx+nw/2,ny+nh/2)
        mask = cv2.imread('Image/doggy_nose.png',cv2.IMREAD_UNCHANGED)
        mask=cv2.resize(mask,(nw,nh))
        mask_gray = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
        _,nose_mask=cv2.threshold(mask_gray,25,255,cv2.THRESH_BINARY_INV)
        nose_area=img[top_left[1]:top_left[1]+nh,top_left[0]:top_left[0]+ nw]
        nose_mask_area=cv2.bitwise_and(nose_area,nose_area,mask=nose_mask)
        final_nose=cv2.add(nose_mask_area,mask)
        img[top_left[1]:top_left[1]+nh,top_left[0]:top_left[0]+ nw]=final_nose
    cv2.imshow("Doggy Nose Filter",img)
    k=cv2.waitKey(10)
    if k == ord('q'):
        break
    if k == ord('s'):
        cv2.imwrite("DoggyNose_Filter.png",img)
cap.release()
cv2.destroyAllWindows()
