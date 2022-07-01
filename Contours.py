import cv2
from cv2 import rectangle
from cv2 import log
from cv2 import resize 
import numpy as np
import os
import matplotlib.pyplot as plt
import math
import random
def sortBBox(listboxmax,listboxmin):
    for i in range(len(listboxmax)):
        for j in range(0,len(listboxmax)-i-1):
            if listboxmax[j][0] > listboxmax[j+1][0]:
                temp = listboxmax[j]
                listboxmax[j] = listboxmax[j+1]
                listboxmax[j+1] = temp
                temp1 = listboxmin[j]
                listboxmin[j] = listboxmin[j+1]
                listboxmin[j+1] = temp1
    return listboxmax, listboxmin
def getBoundingBox(contours):
    listboxmin = []
    listboxmax = []
    for id in range(0,len(contours)):
        listx=[]
        listy=[]
        for coor in contours[id]:
            listx.append(coor[0][0])
            listy.append(coor[0][1])
        minx = np.min(listx)
        miny = np.min(listy)
        maxx = np.max(listx)
        maxy = np.max(listy)
        listboxmin.append((minx,miny))
        listboxmax.append((maxx,maxy))
    return listboxmax, listboxmin
def drawBBox(img,coor):
    img1 = img.copy()
    for i in range(len(coor)):
        cv2.rectangle(img1,coor[i][0],coor[i][1],(255,0,0),1)
    return img1
def drawBBox2(img,coor):
    cv2.rectangle(img,(coor[0],coor[1]),(coor[2],coor[3]),(255,0,0),1)
    return img
def convertColorToWhiteColor(image, threshold_Green_min = 80,threshold_Blue_min = 150,threshold_Red_min = 150, ratio=1.1):

    height,width = image.shape[0],image.shape[1]
    
    for loop1 in range(height):
        for loop2 in range(width):
            r,g,b = image[loop1,loop2]
            if (g/(r+1) > ratio and g > threshold_Green_min and g>b) :
                image[loop1,loop2] = 255,255,255
            if (r/(g+1) > ratio and r > threshold_Red_min and r>b) :
                image[loop1,loop2] = 255,255,255
    return image
def delLine(image):
    maxValue = 255*image.shape[1]
    height,width = image.shape[0],image.shape[1]
    sumRow = np.sum(image,axis=1)  
    for loop1 in range(len(sumRow)):
        if int(sumRow[loop1]) > maxValue*0.5:
            for i in range(width):
               image[loop1][i] = 0
               if loop1 > 1:
                   image[loop1-1][i] = 0
               if loop1 < height-2:
                   image[loop1+1][i] = 0
    return image
def getAvg(listboxmax,listboxmin):
    area = []
    areaHeight = []
    areaWidth = []
    for i in range(len(listboxmax)):
        w = listboxmax[i][0] - listboxmin[i][0]
        h = listboxmax[i][1] - listboxmin[i][1]
        areaHeight.append(float(h))
        areaWidth.append(float(w))
        area.append(w*h)
    avgAreaHeight= np.average(areaHeight)
    avgAreaWidth= np.average(areaWidth)
    avgArea = np.average(area)
    return avgAreaWidth, avgAreaHeight, avgArea
def getAvgFromList(listChar):
    avgW = []
    avgH = []
    avgA = []
    for i in range(len(listChar)):
        avgH.append(listChar[i][1][1]-listChar[i][0][1])
        avgW.append(listChar[i][1][0]-listChar[i][0][0])
        avgA.append((listChar[i][1][0]-listChar[i][0][0])*(listChar[i][1][1]-listChar[i][0][1]))
    return np.average(avgW),np.average(avgH),np.average(avgA)
def avgDistanceChar(listboxmax,listboxmin):
    distance = []
    for i in range(len(listboxmax)-1):
        distance.append(listboxmax[i+1][0] - listboxmin[i][0])
    return np.average(distance)
def avgMaxy(listboxmax):
    lsMaxy = []
    for i in listboxmax:
        lsMaxy.append(i[1])
    return np.average(lsMaxy)
def avgMiny(listboxmin):
    lsMaxy = []
    for i in listboxmin:
        lsMaxy.append(i[1])
    return np.average(lsMaxy)
def centroids(listboxmax,listboxmin):
    x = []
    y = []
    for i in range(len(listboxmax)):
        x.append((listboxmin[i][0] + listboxmax[i][0])/2)
        y.append((listboxmin[i][1] + listboxmax[i][1])/2)
    return (int(np.average(x)),int(np.average(y)))
def centroidscoor(coorXY1,coorXY2):
    x = ((coorXY1[0] + coorXY2[0])/2)
    y = ((coorXY1[1] + coorXY2[1])/2)
    return (int(x),int(y))
def removeBadContours(image,listboxmax1,listboxmin1):
    goodBBox = []
    multiChar = []
    listChar = []
    listboxmax1,listboxmin1 = sortBBox(listboxmax1,listboxmin1)
    avgAreaWidth, avgAreaHeight, avgArea = getAvg(listboxmax1,listboxmin1)
    listBoxMaxFilterSmall = []
    listBoxMinFilterSmall = []
    # listboxmax = []
    # listboxmin = []
    listCharFilterByCentroid = []
    listCentroidy = []
    listMaxY = []
    listCharFilterByDistance = []
    listPhrase = []
    listCountPharse = []
    temp = []
    countChar = 0
    #Delete smal contours
    for i in range(len(listboxmax1)):
        w = listboxmax1[i][0] - listboxmin1[i][0]
        h = listboxmax1[i][1] - listboxmin1[i][1]
        if h * w >  0.2*avgArea  and w > 0.1*avgAreaWidth and h > 0.15* avgAreaHeight: 
            listBoxMaxFilterSmall.append((listboxmax1[i][0],listboxmax1[i][1]))
            listBoxMinFilterSmall.append((listboxmin1[i][0],listboxmin1[i][1]))
    avgAreaWidth, avgAreaHeight, avgArea = getAvg(listBoxMaxFilterSmall,listBoxMinFilterSmall)
    # Filer char from serialNum
    for i in range(len(listBoxMaxFilterSmall)):
        w = listBoxMaxFilterSmall[i][0] - listBoxMinFilterSmall[i][0]
        h = listBoxMaxFilterSmall[i][1] - listBoxMinFilterSmall[i][1]
        # Normal number
        if h > 0.6*avgAreaHeight and h * w >  0.5*avgArea : 
            goodBBox.append([listBoxMinFilterSmall[i],listBoxMaxFilterSmall[i]])
        #Number 1
        elif h > 0.6*avgAreaHeight and h > 1.5*w and w*h > 0.15*avgArea: 
            goodBBox.append([listBoxMinFilterSmall[i],listBoxMaxFilterSmall[i]])
        #Number 0 small
        elif w-h<3 and w*h > 0.2*avgArea: 
            goodBBox.append([listBoxMinFilterSmall[i],listBoxMaxFilterSmall[i]])
    # Filter serial number from goodBBox
    for i in range(len(goodBBox)): 
        w = goodBBox[i][1][0] - goodBBox[i][0][0] 
        h = goodBBox[i][1][1] - goodBBox[i][0][1] 
        #Check multi char
        if w >1.25*h and h * w > 1.4*avgArea or w >1.3*h:
            listChar.append([(goodBBox[i][0][0],goodBBox[i][0][1]),(goodBBox[i][1][0],goodBBox[i][1][1])])
            listCentroidy.append(int((goodBBox[i][0][1]+goodBBox[i][1][1])/2))
        # One char
        else:
            listChar.append([(goodBBox[i][0][0],goodBBox[i][0][1]),(goodBBox[i][1][0],goodBBox[i][1][1])])
            listCentroidy.append(int((goodBBox[i][0][1] + goodBBox[i][1][1])/2))
    avgCentroidy = np.average(listCentroidy)
    # Filter by centroid
    for i in range(len(listChar)):
        centroidtemp = centroidscoor(listChar[i][1],listChar[i][0])
        if abs(centroidtemp[1]-avgCentroidy) < 15 and listChar[i][1][1] + 3 > avgCentroidy:
            listCharFilterByCentroid.append(listChar[i])
            listMaxY.append(listChar[i][1][1])
    avgMaxy = np.average(listMaxY)
    listCharFilterByAvgMaxy = []
    # Filter by avg coor(y)
    for i in range(len(listCharFilterByCentroid)):
        if  listCharFilterByCentroid[i][1][1] + 12 > avgMaxy:
            listCharFilterByAvgMaxy.append(listCharFilterByCentroid[i])
    avgAreaWidth, avgAreaHeight, avgArea = getAvgFromList(listCharFilterByAvgMaxy)

    # Filter by distance 
    for i in range(len(listCharFilterByAvgMaxy)-1):
        w = listCharFilterByAvgMaxy[i][1][0] - listCharFilterByAvgMaxy[i][0][0]
        centroid = centroidscoor(listCharFilterByAvgMaxy[i][1],listCharFilterByAvgMaxy[i][0])
        centroid1 = centroidscoor(listCharFilterByAvgMaxy[i+1][1],listCharFilterByAvgMaxy[i+1][0])
        dist = math.sqrt((centroid[0] - centroid1[0])**2 + (centroid[1] - centroid[1])**2)
        if dist < 3.2*avgAreaHeight:
            temp.append(listCharFilterByAvgMaxy[i])
            countChar +=1
            if i + 2 == len(listCharFilterByAvgMaxy):
                temp.append(listCharFilterByAvgMaxy[i+1])
                listPhrase.append(temp)
                countChar +=1
                listCountPharse.append(countChar)
        else:
            temp.append(listCharFilterByAvgMaxy[i])
            listPhrase.append(temp)
            countChar +=1
            listCountPharse.append(countChar)
            countChar = 0
            temp = []
    for i in range(len(listPhrase)):
        listCharFilterByDistance.append(listPhrase[np.argmax(listCountPharse)])
    listCharFilterByDistance = listCharFilterByDistance[0]
    # Split multichar 
    avgAreaWidth, avgAreaHeight, avgArea = getAvgFromList(listCharFilterByDistance)
    listChar = []
    cooryMax = []
    for i in range(len(listCharFilterByDistance)):
        w = listCharFilterByDistance[i][1][0] - listCharFilterByDistance[i][0][0] 
        h = listCharFilterByDistance[i][1][1] - listCharFilterByDistance[i][0][1]
        #Check multiChar
        if w >1.2*h and h * w > 1.3*avgArea or w >1.47*h:
            multiChar = [listCharFilterByDistance[i][0],listCharFilterByDistance[i][1]]
            imgcp = image.copy()
            imgcp = imgcp[multiChar[0][1]:multiChar[1][1],multiChar[0][0]:multiChar[1][0]]
            # Two char in one box
            if w > 0.5*avgAreaWidth and w < 2.5*avgAreaWidth: 
                imgcp2 = image[multiChar[0][1]:multiChar[1][1],multiChar[0][0]+int(w/2):multiChar[1][0]]
                contourss,hierachy=cv2.findContours(imgcp2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                #Area by pixels
                # bigArea = 0
                # for row in range(imgcp2.shape[0]):
                #     for col in range(imgcp2.shape[1]):
                #         if imgcp2[row][col] > 240:
                #             bigArea+=1
                #Area by contours
                # idMax = 0
                # for idcontourss in range(len(contourss)):
                #     if len(contourss[idcontourss]) > len(contourss[idMax]):
                #         idMax = idcontourss
                # area = cv2.contourArea(contourss[idMax])
                # bigArea = 0
                # for idcontourss in range(len(contourss)):
                #     bigArea += cv2.contourArea(contourss[idcontourss])
                #Area by bbox
                bboxmax,bboxmin = getBoundingBox(contourss)
                bbox = []
                for k in range(len(bboxmax)):
                    bbox.append([bboxmin[k],bboxmax[k]])
                    bigArea = 0
                for l in bbox:
                    w1 = l[1][0] - l[0][0]
                    h1 = l[1][1] - l[0][1]
                    if w1*h1 > bigArea:
                        bigArea = w1*h1
                if bigArea*2 > (w/2)*h:
                # if bigArea > 0.2*(w/2)*h:
                    listChar.append([(multiChar[0][0],multiChar[0][1]),(multiChar[1][0]-int(w/2),multiChar[1][1])])
                    listChar.append([(multiChar[0][0]+int(w/2),multiChar[0][1]),(multiChar[1][0],multiChar[1][1])])
                    cooryMax.append(multiChar[1][1])
                    cooryMax.append(multiChar[1][1])
                else:
                    listChar.append([(multiChar[0][0],multiChar[0][1]),(multiChar[1][0],multiChar[1][1])])
                    cooryMax.append(multiChar[1][1])
            # Three char in one box
            elif w < 4*avgAreaWidth and w >= 2.5*avgAreaWidth:
                listChar.append([(multiChar[0][0],multiChar[0][1]),(multiChar[1][0]-int(w/3)*2,multiChar[1][1])])
                listChar.append([(multiChar[0][0]+int(w/3),multiChar[0][1]),(multiChar[1][0]-int(w/3),multiChar[1][1])])
                listChar.append([(multiChar[1][0]-int(w/3),multiChar[0][1]),(multiChar[1][0],multiChar[1][1])])
                cooryMax.append(multiChar[1][1])
                cooryMax.append(multiChar[1][1])
                cooryMax.append(multiChar[1][1])
        # One char
        else:
            listChar.append([(listCharFilterByDistance[i][0][0],listCharFilterByDistance[i][0][1]),(listCharFilterByDistance[i][1][0],listCharFilterByDistance[i][1][1])])
            cooryMax.append(listCharFilterByDistance[i][1][1])
    avgAreaWidth, avgAreaHeight, avgArea = getAvgFromList(listChar)
    # Filter char "No"
    avgCooryMax = np.average(cooryMax)
    rs = []
    for i in range(len(listChar)):
        # if listChar[i][1][1] + 7 > avgCooryMax:
        if listChar[i][1][1] + avgAreaHeight*0.4 > avgCooryMax:
            rs.append(listChar[i])
    return rs
def resize_image_min( image,input_size=128):
	height,width= image.shape[:2]
	scale_1 = float(input_size/ width)
	scale_2 = float(input_size/ height)
	scale = max(scale_1, scale_2)
	width= int(width*scale)
	height=int(height*scale)
	image= cv2.resize(image,(width,height))
	return image
def splitCharFromSerialNo(image):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    SerialNo = image.copy()
    SerialGray = convertColorToWhiteColor(SerialNo)

    SerialGray = cv2.cvtColor(SerialGray, cv2.COLOR_BGR2GRAY)
    # Inverse 
    m, dev = cv2.meanStdDev(SerialGray)
    ret, thresh = cv2.threshold(SerialGray, m[0][0] - 0.5*dev[0][0], 255, cv2.THRESH_BINARY_INV)
    thresh = delLine(thresh)
    # Padding
    # thresh = padding(thresh)
    
    # Finding countours
    contours,hierachy=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    # plt.imshow(image)
    # plt.show()
    listboxmax,listboxmin = getBoundingBox(contours)
    listChar = removeBadContours(thresh,listboxmax,listboxmin)
    return listChar
def splitCharFromForm(image):
    # image = resize_image_min(image,1280)
    # SerialNo = image[box[1]:box[3],box[0]:box[2]]
    listChar = splitCharFromSerialNo(image)
    return listChar

"""---------------- InOut Area ---------------------"""
def getBBoxFromInOut(image,listboxmax,listboxmin,areaRatio):
    listboxmax1,listboxmin1 = sortBBox(listboxmax,listboxmin)
    listBoxFilterSmall = []
    h,w = image.shape[:2]
    listBBoxChar = [0, 0 , w, h]
    coorXmin = []
    coorXmax = []
    coorYmax = []
    coorYmin = []
    #Delete smal bbox
    for i in range(len(listboxmax)):
        w = listboxmax1[i][0] - listboxmin1[i][0]
        h = listboxmax1[i][1] - listboxmin1[i][1]
        if (w*h > areaRatio[0]*image.shape[0]*image.shape[1] and w > 3*h  and w < 10*h) or (w*h >areaRatio[1]*image.shape[0]*image.shape[1] and w < 5*h):
            sum=0
            for row in range(listboxmin1[i][1],listboxmax1[i][1]):
                for col in range(listboxmin1[i][0],listboxmax1[i][0]):
                    if image[row][col] == 255:
                        sum+=1
            if sum >0.2*w*h:
                listBoxFilterSmall.append((listboxmin[i],listboxmax[i]))
                coorXmin.append(listboxmin1[i][0])
                coorXmax.append(listboxmax1[i][0])
                coorYmax.append(listboxmax1[i][1])
                coorYmin.append(listboxmin1[i][1])
    ret = False
    # imgBox = drawBBox (image,listBoxFilterSmall)
    # cv2.imwrite('/home/anlab/ANLAB/SerialPJ/projects/SerialPJ/results/img.jpg',imgBox)
    #Concatenate char
    if len(listBoxFilterSmall)!= 0:
        w = np.max(coorXmax) - np.min(coorXmin)
        h = np.max(coorYmax) - np.min(coorYmin)
        s = w*h
        # if w > 1.5*h and s > areaRatio[2]*image.shape[0]*image.shape[1]:
        if w > 0.7*h and s > areaRatio[2]*image.shape[0]*image.shape[1]:
            listBBoxChar =[int(np.min(coorXmin)),int(np.min(coorYmin)),int(np.max(coorXmax)),int(np.max(coorYmax))]
            ret= True
    # print(ret)
    return ret, listBBoxChar
    
def getInfo(image, threshold = 0.17,areaRatio=[0.005,0.04,0.15]):
    imageOri = image.copy()
    image = resize_image_min(image,60)
    scale = imageOri.shape[0]/image.shape[0]
    # cv2.imwrite('/home/anlab/ANLAB/SerialPJ/projects/SerialPJ/results/'+basename,image)
    # box_contruction: areaRatio=[0.005,0.01,0.02]
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    InOut = image.copy()
    InOutGray = convertColorToWhiteColor(InOut)
    InOutGray = cv2.cvtColor(InOutGray, cv2.COLOR_BGR2GRAY)
    m, dev = cv2.meanStdDev(InOutGray)    
    ret, thresh = cv2.threshold(InOutGray, m[0][0] + 0.02*dev[0][0], 255, cv2.THRESH_BINARY_INV)
    thresh = delLine(thresh)
    contours,hierachy=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    listboxmax,listboxmin = getBoundingBox(contours)
    
    ret, bounding_box = getBBoxFromInOut(thresh,listboxmax,listboxmin,areaRatio)
    h, w = image.shape[:2]
    if ret :
        est = max(1, int(0.01*(bounding_box[3] - bounding_box[1])))
        xmin = max(0 , bounding_box[0]-est)
        ymin = max(0 , bounding_box[1] -est)
        xmax = min(w , bounding_box[2] + est)
        ymax = min(h , bounding_box[3] + est)   
        bounding_box = [xmin, ymin, xmax, ymax]
        imcrop = thresh[ymin:ymax , xmin:xmax]
        m = cv2.mean(imcrop)
        if m[0]/255.0 < threshold :
            ret = False
        # cv2.imwrite('img1.jpg', imcrop)
        # print("mean" , m[0]/255.0)
        bounding_box= [int(bounding_box[0]*scale),int(bounding_box[1]*scale),int(bounding_box[2]*scale),int(bounding_box[3]*scale)]
    return ret, bounding_box
# def splitBBboxFromElectricMotor(image,box=[1860,1145,2045,1215]):
#     sizeImg=[1280,2308]
#     image = cv2.resize(image,sizeImg)
#     SerialNo = image[box[1]:box[3],box[0]:box[2]]
#     check, listChar = splitCharElectricMotor(image)
#     return image, check, listChar
# cv2.waitKey(0)
# cv2.destroyAllWindows()

