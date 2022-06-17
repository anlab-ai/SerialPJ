from queue import Empty
import cv2
from cv2 import rectangle 
import numpy as np
import os
import matplotlib.pyplot as plt
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

def padding(bin):
    #bottom
    for i in range(bin.shape[1]):
        # for j in range(num):
        bin[bin.shape[0]-2][i]=0
        bin[bin.shape[0]-1][i]=0
    #top
    for i in range(bin.shape[1]):
        bin[0][i]=0
        bin[1][i]=0
    #left
    for i in range(bin.shape[0]):
        bin[i][bin.shape[1]-2]=0
        bin[i][bin.shape[1]-1]=0
    #right
    for i in range(bin.shape[0]):
        bin[i][0]=0
        bin[i][1]=0
    return bin
def coverGreenColorToWhiteColor(image):
    height,width = image.shape[0],image.shape[1]
    for loop1 in range(height):
        for loop2 in range(width):
            r,g,b = image[loop1,loop2]
            if g > 1+r and g>1+b:
                image[loop1,loop2] = 255,255,255
    return image
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
def drawBBox1(img,listboxmax,listboxmin):
    for i in range(len(listboxmax)):
        cv2.rectangle(img,listboxmin[i],listboxmax[i],(255,0,0),1)
        plt.imshow(img)
        plt.show()
    return img
def drawBBox(img,coor):
    for i in range(len(coor)):
        cv2.rectangle(img,coor[i][0],coor[i][1],(255,0,0),1)
    return img
def coverColorToWhiteColor(image):
    height,width = image.shape[0],image.shape[1]
    for loop1 in range(height):
        for loop2 in range(width):
            r,g,b = image[loop1,loop2]
            if g -5 > r and g >= b:
                image[loop1,loop2] = 255,255,255
            elif g -10 > r and g + 1 <= b:
                image[loop1,loop2] = 255,255,255
            elif r >= g + 10 and r >= 10 + b:
                image[loop1,loop2] = 255,255,255
    return image
def coverColorToWhiteColorMX(image):
    height,width = image.shape[0],image.shape[1]
    for loop1 in range(height):
        for loop2 in range(width):
            r,g,b = image[loop1,loop2]
            if g > r +10 and g > b + 10:
                image[loop1,loop2] = 255,255,255
    return image    
def delLine(image):
    maxValue = 255*image.shape[1]
    height,width = image.shape[0],image.shape[1]
    sumRow = np.sum(image,axis=1)  
    for loop1 in range(len(sumRow)):
        if int(sumRow[loop1]) > maxValue*0.7:
            for i in range(width):
               image[loop1][i] = 0
               if loop1 > 1:
                   image[loop1-1][i] = 0
                   image[loop1-2][i] = 0
               if loop1 < height-2:
                   image[loop1+1][i] = 0
                   image[loop1+2][i] = 0
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
def avgDistanceChar(listboxmax,listboxmin):
    distance = []
    for i in range(len(listboxmax)-1):
        distance.append(listboxmax[i+1][0] - listboxmin[i][0])
    return np.average(distance)

"""-----------------------API NEW-------------------------------"""
def removeBadContours(image,listboxmax1,listboxmin1):
    bettetContours = []
    multiChar = []
    listChar = []
    listboxmax1,listboxmin1 = sortBBox(listboxmax1,listboxmin1)
    avgAreaWidth, avgAreaHeight, avgArea = getAvg(listboxmax1,listboxmin1)
    listboxmax2 = []
    listboxmin2 = []
    listboxmax = []
    listboxmin = []
    for i in range(len(listboxmax1)):
        w = listboxmax1[i][0] - listboxmin1[i][0]
        h = listboxmax1[i][1] - listboxmin1[i][1]
        # if h * w >  0.03*avgArea  and w > 0.1*avgAreaWidth and h > 0.1* avgAreaHeight: 
        listboxmax2.append((listboxmax1[i][0],listboxmax1[i][1]))
        listboxmin2.append((listboxmin1[i][0],listboxmin1[i][1]))
    avgAreaWidth, avgAreaHeight, avgArea = getAvg(listboxmax2,listboxmin2)
    avgDistance = avgDistanceChar(listboxmax2,listboxmin2)
    for i in range(len(listboxmax2)-1):
        if listboxmin2[i+1][0] - listboxmax2[i][0] < avgDistance: 
            listboxmax.append((listboxmax2[i][0],listboxmax2[i][1]))
            listboxmin.append((listboxmin2[i][0],listboxmin2[i][1]))
            if i + 1 == len(listboxmax2)-1:
                listboxmax.append((listboxmax2[i + 1][0],listboxmax2[i + 1][1]))
                listboxmin.append((listboxmin2[i + 1][0],listboxmin2[i + 1][1]))
    avgAreaWidth, avgAreaHeight, avgArea = getAvg(listboxmax,listboxmin)
    for i in range(len(listboxmax)):
        w = listboxmax[i][0] - listboxmin[i][0]
        h = listboxmax[i][1] - listboxmin[i][1]
        # for file LK
        # Normal number
        if h > 0.65*avgAreaHeight and h * w >  0.5*avgArea: 
            bettetContours.append([listboxmin[i],listboxmax[i]])
        #Number 1
        elif h > 0.6*avgAreaHeight and h > 1.5*w and w*h > 0.2*avgArea: 
            bettetContours.append([listboxmin[i],listboxmax[i]])
        #Number 0 small
        elif abs(w-h)<5 and w*h > 0.2*avgArea: 
            bettetContours.append([listboxmin[i],listboxmax[i]])
        #for file MX
        # if h > avgAreaHeight/1.5 and h * w > 1.2 * avgArea: 
        #     bettetContours.append([listboxmin[i],listboxmax[i]])
        # elif w < h*2.2 and h > avgAreaHeight/2.5 and h * w > 0.5*avgArea:
        #     bettetContours.append([listboxmin[i],listboxmax[i]])
    id = 0
    for i in range(len(bettetContours)): 
        w = bettetContours[i][1][0] - bettetContours[i][0][0] 
        h = bettetContours[i][1][1] - bettetContours[i][0][1] 
        if w >1.25*h and h * w > avgArea or w >1.8*h :
            multiChar = [bettetContours[i][0],bettetContours[i][1]]
            imgcp = image
            imgcp = imgcp[multiChar[0][1]:multiChar[1][1],multiChar[0][0]:multiChar[1][0]]
            w = multiChar[1][0] - multiChar[0][0] 
            h = multiChar[1][1] - multiChar[0][1] 
            if w > 0.5*avgAreaWidth and w < 2.5*avgAreaWidth :
                imgcp2 = image[multiChar[0][1]:multiChar[1][1],multiChar[0][0]+int(w/2):multiChar[1][0]]
                imgcp2 = padding(imgcp2)
                contourss,hierachy=cv2.findContours(imgcp2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                bboxmax,bboxmin = getBoundingBox(contourss)
                bbox = []
                for k in range(len(bboxmax)):
                    bbox.append([bboxmin[k],bboxmax[k]])
                bigArea = 0
                for l in bbox:
                    w1 = l[1][0] - l[0][0]
                    h1 = l[1][1] - l[0][1]
                    if w1*h1 >bigArea:
                        bigArea = w1*h1
                if 2*bigArea > (w/2)*h:
                    listChar.append([(multiChar[0][0]-2,multiChar[0][1]-2),(multiChar[1][0]-int(w/2)+2,multiChar[1][1]+2)])
                    listChar.append([(multiChar[0][0]+int(w/2)-2,multiChar[0][1]-2),(multiChar[1][0]+2,multiChar[1][1]+2)])
                else:
                    listChar.append([(multiChar[0][0]-2,multiChar[0][1]-2),(multiChar[1][0]+2,multiChar[1][1]+2)])
            elif w < 4*avgAreaWidth and w >= 2.5*avgAreaWidth:
                listChar.append([(multiChar[0][0]-2,multiChar[0][1]-2),(multiChar[1][0]-int(w/3)*2+2,multiChar[1][1]+2)])
                listChar.append([(multiChar[0][0]+int(w/3)-2,multiChar[0][1]-2),(multiChar[1][0]-int(w/3)+2,multiChar[1][1]+2)])
                listChar.append([(multiChar[1][0]-int(w/3)-2,multiChar[0][1]-2),(multiChar[1][0]+2,multiChar[1][1]+2)])
            
        else:
            listChar.append([(bettetContours[i][0][0],bettetContours[i][0][1]),(bettetContours[i][1][0],bettetContours[i][1][1])])
    return listChar
def splitCharFromSerialNo(image):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    SerialNo = image
    SerialGray = coverColorToWhiteColor(SerialNo)
    SerialGray = cv2.cvtColor(SerialGray, cv2.COLOR_BGR2GRAY)
    # Inverse 
    m, dev = cv2.meanStdDev(SerialGray)
    ret, thresh = cv2.threshold(SerialGray, m[0][0] - 0.5*dev[0][0], 255, cv2.THRESH_BINARY_INV)
    thresh = delLine(thresh)
    # Padding
    thresh = padding(thresh)
    
    # Finding countours
    contours,hierachy=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    # plt.imshow(image)
    # plt.show()
    listboxmax,listboxmin = getBoundingBox(contours)
    listChar = removeBadContours(thresh,listboxmax,listboxmin)
    return image, listChar
def splitCharFromForm(image,box=[1600,1380,2034,1460]):
    sizeImg=[2114,2990]
    image = cv2.resize(image,sizeImg)
    SerialNo = image[box[1]:box[3],box[0]:box[2]]
    # print(image.shape)
    image, listChar = splitCharFromSerialNo(SerialNo)
    return image, listChar
# image = cv2.imread('input/MFG No.032200723 LK-11VC-02.jpg')
# image,listChar = splitCharFromForm(image)
# print(listChar)
# image = drawBBox(image,listChar)
# cv2.imwrite("drawing.jpg" ,image )
# plt.imshow(image)
# plt.show()

"""--------------------------------------------------------------"""
# def removeBadContours(filename,oriimg,image,listboxmax1,listboxmin1):
#     bettetContours = []
#     multiChar = []
#     listChar = []
#     listboxmax1,listboxmin1 = sortBBox(listboxmax1,listboxmin1)
#     avgAreaWidth, avgAreaHeight, avgArea = getAvg(listboxmax1,listboxmin1)
#     listboxmax2 = []
#     listboxmin2 = []
#     listboxmax = []
#     listboxmin = []
#     for i in range(len(listboxmax1)):
#         w = listboxmax1[i][0] - listboxmin1[i][0]
#         h = listboxmax1[i][1] - listboxmin1[i][1]
#         # if h * w >  0.03*avgArea  and w > 0.1*avgAreaWidth and h > 0.1* avgAreaHeight: 
#         listboxmax2.append((listboxmax1[i][0],listboxmax1[i][1]))
#         listboxmin2.append((listboxmin1[i][0],listboxmin1[i][1]))
#     avgAreaWidth, avgAreaHeight, avgArea = getAvg(listboxmax2,listboxmin2)
#     avgDistance = avgDistanceChar(listboxmax2,listboxmin2)
#     for i in range(len(listboxmax2)-1):
#         if listboxmin2[i+1][0] - listboxmax2[i][0] < avgDistance: 
#             listboxmax.append((listboxmax2[i][0],listboxmax2[i][1]))
#             listboxmin.append((listboxmin2[i][0],listboxmin2[i][1]))
#             if i + 1 == len(listboxmax2)-1:
#                 listboxmax.append((listboxmax2[i + 1][0],listboxmax2[i + 1][1]))
#                 listboxmin.append((listboxmin2[i + 1][0],listboxmin2[i + 1][1]))
#     avgAreaWidth, avgAreaHeight, avgArea = getAvg(listboxmax,listboxmin)
#     for i in range(len(listboxmax)):
#         w = listboxmax[i][0] - listboxmin[i][0]
#         h = listboxmax[i][1] - listboxmin[i][1]
#         # for file LK
#         # Normal number
#         if h > 0.65*avgAreaHeight and h * w >  0.5*avgArea: 
#             bettetContours.append([listboxmin[i],listboxmax[i]])
#         #Number 1
#         elif h > 0.6*avgAreaHeight and h > 1.5*w and w*h > 0.2*avgArea: 
#             bettetContours.append([listboxmin[i],listboxmax[i]])
#         #Number 0 small
#         elif abs(w-h)<5 and w*h > 0.2*avgArea: 
#             bettetContours.append([listboxmin[i],listboxmax[i]])
#         #for file MX
#         # if h > avgAreaHeight/1.5 and h * w > 1.2 * avgArea: 
#         #     bettetContours.append([listboxmin[i],listboxmax[i]])
#         # elif w < h*2.2 and h > avgAreaHeight/2.5 and h * w > 0.5*avgArea:
#         #     bettetContours.append([listboxmin[i],listboxmax[i]])
#     id = 0
#     for i in range(len(bettetContours)): 
#         w = bettetContours[i][1][0] - bettetContours[i][0][0] 
#         h = bettetContours[i][1][1] - bettetContours[i][0][1] 
#         if w >1.25*h and h * w > avgArea or w >1.8*h :
#             multiChar = [bettetContours[i][0],bettetContours[i][1]]
#             imgcp = image
#             imgcp = imgcp[multiChar[0][1]:multiChar[1][1],multiChar[0][0]:multiChar[1][0]]
#             w = multiChar[1][0] - multiChar[0][0] 
#             h = multiChar[1][1] - multiChar[0][1] 
#             if w > 0.5*avgAreaWidth and w < 2.5*avgAreaWidth :
#                 imgcp2 = image[multiChar[0][1]:multiChar[1][1],multiChar[0][0]+int(w/2):multiChar[1][0]]
#                 imgcp2 = padding(imgcp2)
#                 contourss,hierachy=cv2.findContours(imgcp2,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#                 bboxmax,bboxmin = getBoundingBox(contourss)
#                 bbox = []
#                 for k in range(len(bboxmax)):
#                     bbox.append([bboxmin[k],bboxmax[k]])
#                 bigArea = 0
#                 for l in bbox:
#                     w1 = l[1][0] - l[0][0]
#                     h1 = l[1][1] - l[0][1]
#                     if w1*h1 >bigArea:
#                         bigArea = w1*h1
#                 if 2*bigArea > (w/2)*h:
#                     img1 = oriimg[multiChar[0][1]-2:multiChar[1][1]+2,multiChar[0][0]-2:multiChar[1][0]-int(w/2)+2]
#                     cv2.imwrite('/home/anlab/ANLAB/VisualCode/Contours/SplitHandWriting/OutPut/ResultCrop/'+folderName+'/'+str(id)+'_'+filename,img1)
#                     id+=1
#                     img2 = oriimg[multiChar[0][1]-2:multiChar[1][1]+2,multiChar[0][0]+int(w/2)-2:multiChar[1][0]+2]
#                     cv2.imwrite('/home/anlab/ANLAB/VisualCode/Contours/SplitHandWriting/OutPut/ResultCrop/'+folderName+'/'+str(id)+'_'+filename,img2)
#                     id+=1
#                     listChar.append([(multiChar[0][0]-2,multiChar[0][1]-2),(multiChar[1][0]-int(w/2)+2,multiChar[1][1]+2)])
#                     listChar.append([(multiChar[0][0]+int(w/2)-2,multiChar[0][1]-2),(multiChar[1][0]+2,multiChar[1][1]+2)])
#                 else:
#                     listChar.append([(multiChar[0][0]-2,multiChar[0][1]-2),(multiChar[1][0]+2,multiChar[1][1]+2)])
#                     img = oriimg[multiChar[0][1]-2:multiChar[1][1]+2,multiChar[0][0]-2:multiChar[1][0]+2]
#                     cv2.imwrite('/home/anlab/ANLAB/VisualCode/Contours/SplitHandWriting/OutPut/ResultCrop/'+folderName+'/'+str(id)+'_'+filename,img)
#                     id+=1
#             elif w < 4*avgAreaWidth and w >= 2.5*avgAreaWidth:
#                 listChar.append([(multiChar[0][0]-2,multiChar[0][1]-2),(multiChar[1][0]-int(w/3)*2+2,multiChar[1][1]+2)])
#                 listChar.append([(multiChar[0][0]+int(w/3)-2,multiChar[0][1]-2),(multiChar[1][0]-int(w/3)+2,multiChar[1][1]+2)])
#                 listChar.append([(multiChar[1][0]-int(w/3)-2,multiChar[0][1]-2),(multiChar[1][0]+2,multiChar[1][1]+2)])
#                 img1 = oriimg[multiChar[0][1]-2:multiChar[1][1]+2,multiChar[0][0]-2:multiChar[1][0]-int(w/3)*2+2]
#                 cv2.imwrite('/home/anlab/ANLAB/VisualCode/Contours/SplitHandWriting/OutPut/ResultCrop/'+folderName+'/'+str(id)+'_'+filename,img1)
#                 id+=1
#                 img2 = oriimg[multiChar[0][1]-2:multiChar[1][1]+2,multiChar[0][0]+int(w/3)-2:multiChar[1][0]-int(w/3)+2]
#                 cv2.imwrite('/home/anlab/ANLAB/VisualCode/Contours/SplitHandWriting/OutPut/ResultCrop/'+folderName+'/'+str(id)+'_'+filename,img2)
#                 id+=1
#                 img3 = oriimg[multiChar[0][1]-2:multiChar[1][1]+2,multiChar[1][0]-int(w/3)-2:multiChar[1][0]+2]
#                 cv2.imwrite('/home/anlab/ANLAB/VisualCode/Contours/SplitHandWriting/OutPut/ResultCrop/'+folderName+'/'+str(id)+'_'+filename,img3)
#                 id+=1
            
#         else:
#             listChar.append([(bettetContours[i][0][0],bettetContours[i][0][1]),(bettetContours[i][1][0],bettetContours[i][1][1])])
#             img = oriimg[bettetContours[i][0][1]:bettetContours[i][1][1],bettetContours[i][0][0]:bettetContours[i][1][0]]
#             cv2.imwrite('/home/anlab/ANLAB/VisualCode/Contours/SplitHandWriting/OutPut/ResultCrop/'+folderName+'/'+str(id)+'_'+filename,img)
#             id+=1
#     return bettetContours
# def splitCharFromSerialNo(image,filename):
#     image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#     SerialNo = image
#     SerialGray = coverColorToWhiteColorMX(SerialNo)
#     SerialGray = cv2.cvtColor(SerialGray, cv2.COLOR_BGR2GRAY)
#     # Inverse 
#     m, dev = cv2.meanStdDev(SerialGray)
#     ret, thresh = cv2.threshold(SerialGray, m[0][0] - 0.5*dev[0][0], 255, cv2.THRESH_BINARY_INV)
#     thresh = delLine(thresh)
#     # Padding
#     cv2.imwrite('/home/anlab/ANLAB/VisualCode/Contours/SplitHandWriting/OutPut/flag/'+filename,thresh)
#     thresh = padding(thresh)
    
#     # Finding countours
#     contours,hierachy=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#     # cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
#     # plt.imshow(image)
#     # plt.show()
#     listboxmax,listboxmin = getBoundingBox(contours)
#     listChar = removeBadContours(filename,image,thresh,listboxmax,listboxmin)
#     return listChar
# folderName = 'MX'
# for filename in os.listdir('/home/anlab/ANLAB/VisualCode/Contours/SplitHandWriting/TestImage/input-20220616T103931Z-001/input/'+folderName):
#     image = cv2.imread('/home/anlab/ANLAB/VisualCode/Contours/SplitHandWriting/TestImage/input-20220616T103931Z-001/input/'+folderName+'/'+filename)
#     #LK((1600,1380),(2034,1460)) size(2114,2990)
#     #MX((1600,1785),(2000,1850)) size(2114,2990)
#     #MXM((1850,1910),(2340,1985)) size(2480,3508)
#     #SMX((1840,2000),(2290,2080)) size(2480,3508)
#     sizeImg = (2114,2990)
#     x1 = 1600
#     y1 = 1785
#     x2 = 2000
#     y2 = 1850
#     image = cv2.resize(image,sizeImg)
#     SerialNo = image[y1:y2,x1:x2]
#     listChar = splitCharFromSerialNo(SerialNo,filename)
#     img = drawBBox(SerialNo,listChar)
#     cv2.imwrite('/home/anlab/ANLAB/VisualCode/Contours/SplitHandWriting/OutPut/flag/'+filename,img)


cv2.waitKey(0)
cv2.destroyAllWindows()

