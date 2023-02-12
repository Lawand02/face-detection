import cv2 as cv  #include open cv library
def face_detact():
    face_cascade=cv.CascadeClassifier('haarcascade_fullbody.xml')
    eye_cascade=cv.CascadeClassifier('haarcascade_eye.xml')
    # img=cv.imread('test.JPG')
    cap = cv.VideoCapture(0) # run web video
    while cap.isOpened():
        _,img = cap.read()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
            cv.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
            roi_gray=gray[y:y+h,x:x+w]
            roi_color=img[y:y+h,x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        cv.imshow('img',img)
        if cv.waitKey(1)==ord('q'):
            cv.destroyAllWindows()
            break
    cap.release()
    cv.destroyAllWindows()


face_detact()