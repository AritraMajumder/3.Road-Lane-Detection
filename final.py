import matplotlib.pyplot as plt
import numpy as np
import cv2

def return_edges(img):
    rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    grey = cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(5,5),sigmaX=0,sigmaY=None)
    edges = cv2.Canny(img,100,150)
    return edges

def mask(image):
    height, width = image.shape
    triangle = np.array([
                       [(0, height), (int(width/2), int(height*0.3)), (width, height)] #accuracy highly video dependent on video
                       ])
    mask = np.zeros_like(image)
    mask = cv2.fillPoly(mask, triangle, 255)
    mask = cv2.bitwise_and(image, mask)
    return mask

def hough_transform(avg,img):
    black=np.zeros_like(img)
    for i in avg:
        x1,y1,x2,y2=i
        cv2.line(black,(x1,y1),(x2,y2),(255,0,0),5)
    return black

def averager(lines,img):
    left = []
    right =[]
    if lines is not None: 
        for line in lines:      
            x1,y1,x2,y2 = line.reshape(4)        
            param = np.polyfit((x1,x2),(y1,y2),1) #m at 0 c at 1 
            m = param[0]
            c = param[1]

            if(m<0):
                left.append((m,c))
            else:
                right.append((m,c))
    l_avg = np.average(left,axis=0)  
    r_avg = np.average(right,axis=0)

    left_line = coordinator(l_avg,img)
    right_line = coordinator(r_avg,img)
    
    two_lines = [left_line,right_line]
    return two_lines

def coordinator(line_params,img):
    m = line_params[0]
    c = line_params[1]
    y1 = img.shape[0]
    y2 = int(3/5*y1)
    x1 = int((y1-c)/m)
    x2 = int((y2-c)/m)
    
    l = [x1,y1,x2,y2]
    return l

cap = cv2.VideoCapture("r5.mp4")
while(cap.isOpened):
    ret,frame = cap.read()  
    b=return_edges(frame)       #canny edges
    c=mask(b)                   #isolated edges
    lines = cv2.HoughLinesP(c, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    d = averager(lines,frame)   #averaged line coords
    e = hough_transform(d,frame)#averaged lines

    f = cv2.addWeighted(frame,0.8,e,1,1)
    cv2.imshow('frame',f)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
