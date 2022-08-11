import numpy
import cv2
from common import ErrorCode
import imutils
import constant

startDebug = False

def convertBinaryImage(inputImage):
	errCode = ErrorCode.SUCCESS
	gray = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
	# cv2.bitwise_not(gray, gray)
	binary_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 7)
	# m, dev = cv2.meanStdDev(gray)
	# threshold_ = max(50, m[0])
	# threshold_, bw = cv2.threshold(gray, threshold_, 255, cv2.THRESH_BINARY)
	# m, dev = cv2.meanStdDev(gray, m, dev, bw)
	# if (dev[0] > 5):
	# 	threshold_ = m[0] - 0.9*dev[0]
	# 	if (threshold_ < 50):
	# 		threshold_ = 50
	# 	threshold_, bw = cv2.threshold(gray, threshold_, 255, cv2.THRESH_BINARY)
	# binary_image = cv2.bitwise_and(binary_image, bw)
	return errCode, binary_image

# def getAngle(p1, p2, l1, l2):
# 	errCode = common.SUCCESS
# 	t1 = p2 - p1
# 	t2 = l2 - l1
# 	distan1 = math.sqrtf(t1.x *t1.x + t1.y*t1.y)
# 	distan2 = math.sqrtf(t2.x *t2.x + t2.y*t2.y)
# 	cos_angle = float(t1.x*t2.x + t1.y*t2.y) / float(distan1 *distan2)
# 	angle = cv2.acos(cos_angle)
# 	return errCode, angle

# def findVerticalLine(binaryImage):
# 	errCode = common.SUCCESS
	
# 	height, width = binaryImage.shape

# 	draw = cv2.cvtColor(binaryImage, cv2.COLOR_GRAY2RGB)
# 	lines = cv2.HoughLinesP(binaryImage, 2, numpy.pi / 180, 100, height/3, 3)
# 	# map<float, vector<lineInfo>> verticalLine;
# 	verticalLines = []
# 	# print("lines = ", lines)
# 	for line in lines:
# 		# print(line)
# 		infoLine = line[0]
# 		cv2.line(draw,(infoLine[0],infoLine[1]),(infoLine[2],infoLine[3]),(0,0,255),1)
# 		# errCode, angle = getAngle((line[0][0], line[0][1]), (line[0][2], line[0][3]), (0, 0), (0, 50))
# 		# angle = angle*180.0 / numpy.pi
# 		# cv::line(draw, Point(listLine[i][0], listLine[i][1]), Point(listLine[i][2], listLine[i][3]), Scalar(0, 255, 0), 1);
# 		# print("angle: ", angle)
# 		# if ((abs(angle_) < 45 || abs(angle_) > 135))
# 		# {
# 		# 	lineInfo line;
# 		# 	line.line = listLine[i];
# 		# 	if (listLine[i][0] == listLine[i][2]) {
# 		# 		line.a = 1;
# 		# 		line.b = 0;
# 		# 		line.c = -listLine[i][0];
# 		# 	}
# 		# 	else {
# 		# 		line.a = listLine[i][1] - listLine[i][3];
# 		# 		line.b = listLine[i][2] - listLine[i][0];
# 		# 		line.c = -listLine[i][0] * (listLine[i][1] - listLine[i][3]) - listLine[i][1] * (listLine[i][2] - listLine[i][0]);
# 		# 	}
# 		# 	verticalLine[angle_].push_back(line);
# 		# 	//verticalLine.push_back(line);
# 		# 	cv::line(draw, Point(listLine[i][0], listLine[i][1]), Point(listLine[i][2], listLine[i][3]), Scalar(0, 255, 0), 1);
# 		# }
# 	# }
# 	# cv2.imshow("draw", draw)
# 	return errCode, line

# def findHorizontalLine(binaryImage):
# 	errCode = common.SUCCESS
# 	height, width = binaryImage.shape
# 	return errCode

# def findBorderLine(binaryImage):
# 	errCode = common.SUCCESS
# 	height, width = binaryImage.shape

# 	leftImage = binaryImage[0:height, 0:int(width/2)]
# 	# cv2.imshow("leftImage", leftImage)
# 	leftLine = findVerticalLine(leftImage)

# 	# rightImage = binaryImage[0:height, int(width/2):width]
# 	# # cv2.imshow("rightImage", rightImage)
# 	# leftLine = findVerticalLine(rightImage)

# 	# topImage = binaryImage[0:int(height/2), 0:width]
# 	# # cv2.imshow("topImage", topImage)
# 	# topLine = findHorizontalLine(topImage)

# 	# bottomImage = binaryImage[int(height/2):height, 0:width]
# 	# # cv2.imshow("bottomImage", bottomImage)
# 	# bottomLine = findHorizontalLine(bottomImage)
# 	return errCode

def findBoundingPlateByCanny(image):
	errCode = ErrorCode.SUCCESS
	ratio = image.shape[0] / 50.0
	orig = image.copy()
	image = imutils.resize(image, height = 50)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.bilateralFilter(gray, 11, 17, 17)

	edged = cv2.Canny(gray, 30, 200)
	# cv2.imshow("edged", edged)

	cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
	screenCnt = None
	for c in cnts:
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.015 * peri, True)
		if len(approx) == 4:
			screenCnt = approx
			break

	rect = None
	if type(screenCnt) is numpy.ndarray:
		cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 1)
		# cv2.imshow("image", image)
		pts = screenCnt.reshape(4, 2)
		rect = numpy.zeros((4, 2), dtype = "float32")
		s = pts.sum(axis = 1)
		rect[0] = pts[numpy.argmin(s)]
		rect[2] = pts[numpy.argmax(s)]
		diff = numpy.diff(pts, axis = 1)
		rect[1] = pts[numpy.argmin(diff)]
		rect[3] = pts[numpy.argmax(diff)]
		rect *= ratio
	else:
		errCode = ErrorCode.INVALID_DATA

	if errCode == ErrorCode.SUCCESS:
		if abs(rect[0][0] - rect[1][0]) < image.shape[0]/2 or abs(rect[0][1] - rect[2][1]) < image.shape[1]/2:
			errCode = ErrorCode.INVALID_DATA

	return errCode, rect

def getPlateWithBounding(image, boundingPoly):
	errCode = ErrorCode.SUCCESS
	# print ("boundingPoly = ", boundingPoly)
	# pts = boundingPoly.reshape(4, 2)
	# pts = numpy.array([boundingPoly[0][0], boundingPoly[1][0], boundingPoly[2][0], boundingPoly[3][0]])
	# print("pts = ", pts)
	errCode, plate = fourPointTransform(image, boundingPoly)
	# cv2.imshow("Plate", plate)
	return errCode, plate

def fourPointTransform(image, rect):
	errCode = ErrorCode.SUCCESS
	# # expand image
	# height, width, cn = image.shape
	# image = cv2.copyMakeBorder(image, int(height/2), int(height/2), int(width/2), int(width/2), cv2.BORDER_CONSTANT, None, value = 0)
	# cv2.imshow("image", image)
	# for i in range(pts.size):
	# 	index1 = i//2
	# 	index2 = i%2
	# 	if index2 == 0:
	# 		pts[index1][index2] = pts[index1][index2] + width/2
	# 	else:
	# 		pts[index1][index2] = pts[index1][index2] + height/2

	# obtain a consistent order of the points and unpack them
	# individually
	(tl, tr, br, bl) = rect
	# print("rect = ", rect)
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = numpy.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = numpy.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# print("maxWidth = ", maxWidth)
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = numpy.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = numpy.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))

	maxWidth = int(maxWidth*standardPlateSize[1]/maxHeight)
	maxHeight = standardPlateSize[1]
	# print("maxHeight = ", maxHeight)
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = numpy.array([
		[0, 0],
		[maxWidth-1, 0],
		[maxWidth-1, maxHeight-1],
		[0, maxHeight-1]], dtype = "float32")
	# print("dst = ", dst)
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	
	# print("warped.shape = ", warped.shape)
	# return the warped image
	return errCode, warped

def getTextLinesByContour(image):
	errCode = ErrorCode.SUCCESS
	errCode, binaryImage = convertBinaryImage(image)
	lineImgs = []
	binaryLineImgs = []
	if errCode == ErrorCode.SUCCESS:
		img, contours, hierarchy = cv2.findContours(binaryImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		# draw = cv2.cvtColor(binaryImage, cv2.COLOR_GRAY2RGB)
		hierarchy = hierarchy[0]
		boudingBoxs = []
		for i in range(len(contours)):
			if hierarchy[i][3] == -1:
				x,y,w,h = cv2.boundingRect(contours[i])
				boudingBoxs.append([x,y,w,h])
				# draw = cv2.rectangle(draw, (x,y), (x+w,y+h), (0, 0, 255), 1)

		errCode, lines = mergeLineText(binaryImage, boudingBoxs)
	
	if errCode == ErrorCode.SUCCESS:
		appendSize = standardPlateSize[1]*0.02
		heightImg, widthImg, n = image.shape
		for i in range(len(lines)):
			# draw = cv2.rectangle(draw, (line[0],line[1]), (line[0] + line[2],line[1] + line[3]), (0, 0, 255), 1)
			x,y,w,h = lines[i]
			x1 = max(int(x - appendSize*i), 0)
			y1= max(int(y - appendSize*(i+1)), 0)
			x2 = min(int(x + w + appendSize*i), widthImg)
			y2 = min(int(y + h + appendSize*(i+1)), heightImg)
			# lineImgs.append(image[int(y - appendSize*i):int(y + h + appendSize*i), int(x - appendSize*i): int(x + w + appendSize*i)])
			lineImgs.append(image[y1:y2, x1:x2])
			binaryLineImgs.append(binaryImage[y1:y2, x1:x2])

		# cv2.imshow("draw", draw)
	return errCode, lineImgs, binaryLineImgs

def mergeLineText(binaryImage, boudingBoxs):
	errCode = ErrorCode.SUCCESS
	heightImg, widthImg = binaryImage.shape
	lines = []
	if len(boudingBoxs) >= 2:
		for i in range(len(boudingBoxs)):
			for j in range(i + 1, len(boudingBoxs)):
				if boudingBoxs[i][1] > boudingBoxs[j][1]:
					t = boudingBoxs[i]
					boudingBoxs[i] = boudingBoxs[j]
					boudingBoxs[j] = t
				elif (boudingBoxs[i][1] == boudingBoxs[j][1]) and (boudingBoxs[i][3] < boudingBoxs[j][3]):
					t = boudingBoxs[i]
					boudingBoxs[i] = boudingBoxs[j]
					boudingBoxs[j] = t
		
		selectBoundingBox = []
		for i in range(len(boudingBoxs) - 1):
			if boudingBoxs[i][2] <= widthImg*0.02 or boudingBoxs[i][3] >= heightImg*0.8 or boudingBoxs[i][0] < widthImg*0.05:
				continue
			else:
				selectBoundingBox.append(boudingBoxs[i])

		boundingBoxLines = []
		for i in range(len(selectBoundingBox) - 1):
			flag = True
			if (selectBoundingBox[i][3] > heightImg) and (selectBoundingBox[i + 1][3] > 20):
				y_min = max(selectBoundingBox[i][1], selectBoundingBox[i + 1][1])
				y_max = min(selectBoundingBox[i][1] + selectBoundingBox[i][3], selectBoundingBox[i + 1][1] + selectBoundingBox[i + 1][3])
				if (y_max - y_min > heightImg*0.1):
					flag = True
			else:
				flag = True

			if (selectBoundingBox[i][1] + selectBoundingBox[i][3] - selectBoundingBox[i + 1][1]) > -1 and flag:
				x_min = min(selectBoundingBox[i][0], selectBoundingBox[i + 1][0])
				y_min = min(selectBoundingBox[i][1], selectBoundingBox[i + 1][1])
				y_max = max(selectBoundingBox[i][1] + selectBoundingBox[i][3], selectBoundingBox[i + 1][1] + selectBoundingBox[i + 1][3])
				x_max = max(selectBoundingBox[i][0] + selectBoundingBox[i][2], selectBoundingBox[i + 1][0] + selectBoundingBox[i + 1][2])
				selectBoundingBox[i+1] = [x_min, y_min, x_max - x_min, y_max - y_min]
				if (i+1 == len(selectBoundingBox) - 1):
					boundingBoxLines.append(selectBoundingBox[i+1])
			else:
				boundingBoxLines.append(selectBoundingBox[i])

		for i in range(len(boundingBoxLines)):
			if boundingBoxLines[i][2] < heightImg*0.2 or boundingBoxLines[i][3] < heightImg*0.2:
				continue
			else:
				lines.append(boundingBoxLines[i])
		if len(lines) == 0:
			errCode = ErrorCode.INVALID_DATA
	return errCode, lines


def spilitLineImage(lineImage, binaryLineImage):
	errCode = ErrorCode.SUCCESS
	heightImg, widthImg = binaryLineImage.shape
	img, contours, hierarchy = cv2.findContours(binaryLineImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	hierarchy = hierarchy[0]
	x_spilit = 0
	lineImgs = []
	binaryLineImgs = []
	for i in range(len(contours)):
		if hierarchy[i][3] == -1:
			x,y,w,h = cv2.boundingRect(contours[i])
			if h < binaryLineImage.shape[0]*0.7:
				if y > binaryLineImage.shape[0]*0.2 and x < binaryLineImage.shape[1]*0.3:
					# draw = cv2.rectangle(draw, (x,y), (x+w,y+h), (0, 0, 255), 1)
					# cv2.drawContours(binaryLineImage, contours[i], -1, 0)
					if x_spilit < x + w:
						x_spilit = x + w
					# break
				# else:
				# 	cv2.fillPoly(binaryLineImage, pts =[contours[i]], color=0)

	if x_spilit == 0:
		lineImgs.append(lineImage)
		binaryLineImgs.append(binaryLineImage)
	else:
		lineImgs.append(lineImage[int(heightImg*0.2):int(heightImg*0.8), 0:x_spilit])
		lineImgs.append(lineImage[0:heightImg, x_spilit: widthImg])
		binaryLineImgs.append(binaryLineImage[int(heightImg*0.2):int(heightImg*0.8), 0:x_spilit])
		binaryLineImgs.append(binaryLineImage[0:heightImg, x_spilit: widthImg])
	return errCode, lineImgs, binaryLineImgs

def removeHorizontalLineTable(binaryImg, threshold, widthLine):
	errCode = ErrorCode.SUCCESS
	# print(f'number row = {binaryImg}')
	mask = binaryImg
	height,width = mask.shape
	# Apply edge detection method on the image
	# edges = cv2.Canny(binaryImg,50,150,apertureSize = 3)
	# cv2.imshow("edges",edges)
	# This returns an array of r and theta values
	lines = cv2.HoughLines(binaryImg,1,numpy.pi/360, int(width*threshold), min_theta=(numpy.pi/360)*176, max_theta=(numpy.pi/360)*184)
	if lines is None :
		errCode = ErrorCode.INVALID_DATA

	if errCode == ErrorCode.SUCCESS:
		if startDebug == True:
			print(f'len(lines) = {len(lines)}')

		for line in lines:
			for r_theta in line:
				# print(f'r_theta = {r_theta}')
				# print(f'len(r_theta) = {len(r_theta)}')
				r,theta = r_theta
				# Stores the value of cos(theta) in a
				a = numpy.cos(theta)
		
				# Stores the value of sin(theta) in b
				b = numpy.sin(theta)
				
				# x0 stores the value rcos(theta)
				x0 = a*r
				
				# y0 stores the value rsin(theta)
				y0 = b*r
				
				x1 = int(x0 + (0-x0)*(-b))
				y1 = int(y0 + (0-x0)*(a))
				x2 = int(x0 - (width-x0)*(-b))
				y2 = int(y0 - (width-x0)*(a))
				
				# cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
				# (0,0,255) denotes the colour of the line to be
				cv2.line(mask,(x1,y1), (x2,y2), (0), widthLine)
	return errCode, mask

def findBorderHorizontalLine(binaryImg, threshold, isTopLine):
	errCode = ErrorCode.SUCCESS
	mask = binaryImg
	height,width = mask.shape
	# Apply edge detection method on the image
	# edges = cv2.Canny(binaryImg,50,150,apertureSize = 3)
	# cv2.imshow("edges",edges)
	# This returns an array of r and theta values
	lines = cv2.HoughLines(binaryImg,1,numpy.pi/360, int(width*threshold), min_theta=(numpy.pi/360)*176, max_theta=(numpy.pi/360)*184)
	if lines is None :
		errCode = ErrorCode.INVALID_DATA
	y_line = -1
	draw = cv2.cvtColor(binaryImg,cv2.COLOR_GRAY2RGB)
	if errCode == ErrorCode.SUCCESS:
		if startDebug == True:
			print(f'len(lines) = {len(lines)}')
		
		for line in lines:
			for r_theta in line:
				# print(f'r_theta = {r_theta}')
				# print(f'len(r_theta) = {len(r_theta)}')
				r,theta = r_theta
				# Stores the value of cos(theta) in a
				a = numpy.cos(theta)
		
				# Stores the value of sin(theta) in b
				b = numpy.sin(theta)
				
				# x0 stores the value rcos(theta)
				x0 = a*r
				
				# y0 stores the value rsin(theta)
				y0 = b*r
				
				x1 = int(x0 + (0-x0)*(-b))
				y1 = int(y0 + (0-x0)*(a))
				x2 = int(x0 - (width-x0)*(-b))
				y2 = int(y0 - (width-x0)*(a))
				
				# cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
				# (0,0,255) denotes the colour of the line to be
				# if startDebug:
				# 	print(f'x0,y0 = {x0},{y0}')
				# 	print(f'x1,y1 = {x1},{y1}')
				# 	print(f'x2,y2 = {x2},{y2}')
				# if y1 == 8:
				cv2.line(draw,(x1,y1), (x2,y2), (0,0,255), 1)
				
				if isTopLine:
					if y_line == -1:
						y_line = min(y1,y2)
					else:
						y_line = min(y_line, min(y1,y2))
				else:
					if y_line == -1:
						y_line = max(y1,y2)
					else:
						y_line = max(y_line, max(y1,y2))
	if startDebug:
		cv2.imshow("draw", draw)
	return errCode, y_line

def removeVerticalLineTable(binaryImg, threshold, widthLine):
	errCode = ErrorCode.SUCCESS
	# print(f'number row = {binaryImg}')
	mask = binaryImg
	height,width = mask.shape
	# Apply edge detection method on the image
	# edges = cv2.Canny(binaryImg,50,150,apertureSize = 3)
	# cv2.imshow("edges",edges)
	# This returns an array of r and theta values
	lines = cv2.HoughLines(binaryImg,1,numpy.pi/360, int(height*threshold), min_theta=(numpy.pi/360)*(-4), max_theta=(numpy.pi/360)*(4))
	if lines is None :
		if startDebug:
			print(f'removeVerticalLineTable_len(lines) = empty')
		errCode = ErrorCode.INVALID_DATA
	if startDebug:
		draw = cv2.cvtColor(binaryImg,cv2.COLOR_GRAY2RGB)
	if errCode == ErrorCode.SUCCESS:
		if startDebug:
			print(f'removeVerticalLineTable_len(lines) = {len(lines)}')
		for line in lines:
			for r_theta in line:
				# print(f'r_theta = {r_theta}')
				# print(f'len(r_theta) = {len(r_theta)}')
				r,theta = r_theta
				# Stores the value of cos(theta) in a
				a = numpy.cos(theta)
		
				# Stores the value of sin(theta) in b
				b = numpy.sin(theta)
				
				# x0 stores the value rcos(theta)
				x0 = a*r
				
				# y0 stores the value rsin(theta)
				y0 = b*r
				
				# x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
				x1 = int(x0 + 1000*(-b))
				
				# y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
				y1 = int(y0 + 1000*(a))
		
				# x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
				x2 = int(x0 - 1000*(-b))
				
				# y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
				y2 = int(y0 - 1000*(a))
				
				# cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
				# (0,0,255) denotes the colour of the line to be
				cv2.line(mask,(x1,y1), (x2,y2), (0), widthLine)
				if startDebug:
					cv2.line(draw,(x1,y1), (x2,y2), (0,0,255), widthLine)	
		if startDebug:
			cv2.imshow("removeVerticalLineTable_draw", draw)
			cv2.waitKey()
	return errCode, mask

def findSelectionByContour(binaryImg):
	errCode = ErrorCode.SUCCESS
	imgHeight,imgWidth = binaryImg.shape
	selections = []
	contours, hierarchy = cv2.findContours(binaryImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	hierarchy = hierarchy[0]
	for i in range(len(contours)):
		if hierarchy[i][3] == -1 and cv2.contourArea(contours[i]) > (constant.CHAR_HEIGHT + constant.CHAR_WIDTH):
			contourBox = cv2.boundingRect(contours[i])
			if contourBox[3] > constant.CHAR_HEIGHT \
				and contourBox[3]*contourBox[2] > constant.CHAR_WIDTH*constant.CHAR_HEIGHT \
					and contourBox[3]* contourBox[2] < 0.5 *imgWidth * imgHeight \
						and contourBox[3]* contourBox[2] <= 6*constant.CHAR_WIDTH * constant.CHAR_HEIGHT:
				selections.append(contourBox)
			else:
				binaryImg = cv2.drawContours(binaryImg, contours, i, 0, -1)
		else:
			binaryImg = cv2.drawContours(binaryImg, contours, i, 0, -1)

	if startDebug == True:
		print(f'selections = {selections}')
		cv2.imshow("findSelectionByContour_binaryImg", binaryImg)
	return errCode, selections

def filterBackgroundByColor(image, grayImg, thresholds):
	errCode = ErrorCode.SUCCESS
	if startDebug:
		cv2.imshow("filterBackgroundByColor_grayImg", grayImg)
	h,w = grayImg.shape
	for y in range(h):
		for x in range(w):
			b = int(image[y,x,0])
			g = int(image[y,x,1])
			r = int(image[y,x,2])
			# print(f'b,g,r = {b},{g},{r}')
			if ((abs(r - g) - (max(r, g)*0.2) > 0 and g > r and g>thresholds//2) or (abs(r - b) - (max(r, b)*0.2) > 0 and b > r  and b>thresholds//2) or (abs(g - b) - (max(g, b)*0.2) > 0 and max(b, g)>thresholds//2 and (r < b or r < g)))\
  or (r > thresholds and g > thresholds and b > thresholds):
				grayImg[y,x] = 0

	if startDebug:
		cv2.imshow("filterBackgroundByColor_grayImg", grayImg)
	return errCode, grayImg

def getContentArea(binaryImg, sizeKenel):
	errCode = ErrorCode.SUCCESS
	morphKernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(sizeKenel, sizeKenel))
	dst = cv2.morphologyEx(binaryImg, cv2.MORPH_CLOSE, morphKernel2)
	if startDebug:
		cv2.imshow("getContentArea_dst", dst)
	contours, hierarchies = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	if contours is None or len(contours) == 0:
		errCode = ErrorCode.INVALID_DATA
	y_min, x_min = binaryImg.shape
	y_max, x_max = 0, 0
	if errCode == ErrorCode.SUCCESS:
		for i in range(len(contours)):
			box = cv2.boundingRect(contours[i])
			if x_min > box[0]:
				x_min = box[0]
			if y_min > box[1]:
				y_min = box[1]
			if x_max < box[0] + box[2]:
				x_max = box[0] + box[2]
			if y_max < box[1] + box[3]:
				y_max = box[1] + box[3]
		if (x_min > x_max or y_min > y_max):
			x_min = 0
			y_min = 0
			y_max = binaryImg.shape[0]
			x_max = binaryImg.shape[1]
		if startDebug:
			print(f'getContentArea_box: {(x_min,y_min,x_max,y_max)}')
	return errCode, [x_min,y_min,x_max - x_min,y_max - y_min]

def findMainArea(binaryImg, sizeKenel,  minHeightThreshold = 0.33, maxHeightThreshold = 1, ratioThreshold = 3):
	errCode = ErrorCode.SUCCESS
	morphKernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(sizeKenel, sizeKenel))
	dst = cv2.morphologyEx(binaryImg, cv2.MORPH_CLOSE, morphKernel2)
	if startDebug:
		print(f'findMainArea_image_size = {binaryImg.shape}')
		cv2.imshow("findMainArea_dst", dst)
	contours, hierarchies = cv2.findContours(dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	hierarchies = hierarchies[0]
	cnts = []
	for i in range(len(contours)):
		if hierarchies[i][3] == -1:
			cnts.append(contours[i])
	cntsSorted = sorted(cnts, key=lambda x: cv2.boundingRect(x)[0],reverse=False)
	if startDebug:
		print(f'cntsSorted = {cntsSorted}')
	y_min, x_min = binaryImg.shape
	y_max, x_max = 0, 0
	for i in range(len(cntsSorted)):
		box = cv2.boundingRect(cntsSorted[i])
		if startDebug:
			print(f'findMainArea_contour_box = {box}')
		if box[3] < binaryImg.shape[0]*minHeightThreshold \
			or box[3] > binaryImg.shape[0]*maxHeightThreshold  \
			or box[3]/box[2] > ratioThreshold or box[2]/box[3] > ratioThreshold \
			or (x_max > 0 and box[0] - x_max > 5*constant.CHAR_WIDTH):
			continue
		
		if x_min > box[0]:
			x_min = box[0]
		if y_min > box[1]:
			y_min = box[1]
		if x_max < box[0] + box[2]:
			x_max = box[0] + box[2]
		if y_max < box[1] + box[3]:
			y_max = box[1] + box[3]
	if (x_min > x_max or y_min > y_max):
		x_min = 0
		y_min = 0
		y_max = binaryImg.shape[0]
		x_max = binaryImg.shape[1]
	if startDebug:
		print(f'findMainArea_box: {(x_min,y_min,x_max,y_max)}')
	return errCode, [x_min,y_min,x_max - x_min,y_max - y_min]

# def filterLineText(binaryImg):
# 	errCode = ErrorCode.SUCCESS
# 	contours, hierarchies = cv2.findContours(binaryImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# 	hierarchies = hierarchies[0]
# 	count = cv2.countNonZero(binaryImg)
# 	y_min, x_min = binaryImg.shape
# 	y_max, x_max = 0, 0
# 	for i in range(len(contours)):
# 		if hierarchies[i][3] != -1:
# 			continue
# 		box = cv2.boundingRect(contours[i])
# 		print(f'box = {box}')
# 		#remove small contour
# 		if box[3] < binaryImg.shape[0]/2 or abs(box[3]/box[2] - 1) < 3:
# 			cv2.drawContours()
# 	return errCode

def preprocess(img, imgSize):
    ''' resize, transpose and standardization grayscale images '''
    # create target image and copy sample image into it
    
    widthTarget, heightTarget = imgSize 
    height, width = img.shape 
    factor_x = width / widthTarget
    factor_y = height / heightTarget
    factor = max(factor_x, factor_y)
    # scale according to factor
    newSize = (min(widthTarget, int(width / factor)), min(heightTarget, int(height / factor))) 
    #print ('newSize ={}, old size = {}'.format(newSize, img.shape ))
    img = cv2.resize(img, newSize)
    target = numpy.zeros(shape=(heightTarget, widthTarget), dtype='uint8') * 255 #tao ma tran 255 (128,32)
    target[0:newSize[1], 0:newSize[0]] = img #Padding trên hoặc dưới

    #transpose
    img = cv2.transpose(target)
    # standardization
    mean, stddev = cv2.meanStdDev(img)
    mean = mean[0][0]
    stddev = stddev[0][0] # standard deviation
    #print ('mean ={}, stddev = {}'.format(mean, stddev))
    img = img - mean    
    img = img // stddev if stddev > 0 else img 
    #print ('set', set(img.flatten()))
    #img out co shape (128,32)
    return img

