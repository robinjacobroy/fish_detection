''' 
Author: Robin Jacob Roy
Santhom Computing Facility
robinjacobroy1@gmail.com
12-10-2018
'''


import cv2
import numpy as np



def hist(frame):
    
    #convert target from bgr to hsv space
    hsvt = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    #select region of interest as rectangle
    r = cv2.selectROI(frame)
    hsvroi=hsvt[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]

    # calculating object histogram
    roihist = cv2.calcHist([hsvroi],[0, 1], None, [30, 50], [0, 180, 0, 256] )

    # normalize histogram and apply backprojection
    cv2.normalize(roihist,roihist,0,255,cv2.NORM_MINMAX)
    return(roihist)


def backprojection(orig_frame,final_frame,norm_hist):
    orig=orig_frame.copy()
    hsvt = cv2.cvtColor(final_frame,cv2.COLOR_BGR2HSV)
    
    #calulation of Histogram Backprojection
    dst = cv2.calcBackProject([hsvt],[0,1],norm_hist,[0,180,0,256],1)     

    # Now convolute with circular disc
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    cv2.filter2D(dst,-1,disc,dst)

    # Otsu's thresholding on backprojectd image
    ret2,th2 = cv2.threshold(dst,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    im2, contours, hierarchy = cv2.findContours(th2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    #drawing rectangle on larger contours
    for i in range (0,len(contours)):
        area = cv2.contourArea(contours[i])
        if (area>1200):
            rect = cv2.minAreaRect(contours[i])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(orig_frame,[box],-1,(0,255,0),2)
    
    
    
    
    #cv2.drawContours(im2,contours, -1, (255, 255, 255), -1)
    #res = cv2.bitwise_and(orig,orig,mask=im2)
    cv2.imshow("frame_final",orig_frame)
    #cv2.imshow('contoured_fish',res)
    


#Background Subtraction using KNN
def bgsub(cap,hist):
    cap.set(1,10)
    fgbg = cv2.createBackgroundSubtractorKNN()
    while(1):
        
        ret, frame = cap.read()
        if frame is None:
            break
        fgmask = fgbg.apply(frame)
        
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        ret2,opening = cv2.threshold(opening,0,255,cv2.THRESH_BINARY)
        
        
        erosion = cv2.erode(opening,kernel,iterations = 1)
        dilation = cv2.dilate(erosion,kernel,iterations =5 )
        
        out=cv2.bitwise_and(frame,frame,mask=opening)
        
        out1=cv2.bitwise_and(frame,frame,mask=dilation)

        cv2.imshow('BG_Subtracted', out1)
        
        backprojection(frame,out1,hist)
        
        k = cv2.waitKey(30) & 0xff
        
        if k == 27:
            break
    cap.release()

cap = cv2.VideoCapture('/home/robin/output.mp4')        #input static video
cap.set(1,390)                                          #to select Puntius from frame no.390
ret,ref_frame=cap.read()
norm_hist=hist(ref_frame)

bgsub(cap,norm_hist)       

cv2.destroyAllWindows()

