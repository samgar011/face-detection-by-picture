import cv2 as cv


img = cv.imread('images/crowded.jpg')
cv.imshow('insan yuzu tanima', img)

gri = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('yuzler', gri)

def rescaleFrame(frame, scale = 0.75):
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)

    dimensions= (width,height)
    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)




yuz_tanima = cv.CascadeClassifier('haar_face.xml')

yuz1 = yuz_tanima.detectMultiScale(gri, scaleFactor = 1.1, minNeighbors = 3)

print(f'Bulunan Yuzler = {len(yuz1)}')

for (x,y,w,h) in yuz1:
    cv.rectangle(img, (x,y), (x+w,y+h),(0,255,0), thickness=1)


cv.imshow('Belirlenen Yuzler', img)


cv.waitKey(0)