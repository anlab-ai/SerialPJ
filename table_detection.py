
import cv2
import numpy as np
from imutils import contours as cont
from collections import defaultdict
from PIL import ImageFont, ImageDraw, Image
from imutils import paths
import math

import os
class Line():
	def __init__(self, startx, starty, endx, endy):
		self.startx = startx
		self.starty = starty
		self.endx = endx
		self.endy = endy
		
	def __str__(self):
		return 'Line:{},{},{},{}'.format(self.startx, self.starty, self.endx, self.endy)
	def lenx(self):
		return abs(self.startx - self.endx)
	
	def leny(self):
		return abs(self.starty - self.endy)
	
	def toArray(self):
		return [self.startx, self.starty, self.endx, self.endy]

def reDrawLine(img, thshold = 0.5):
	w, h = img.shape[0], img.shape[1]
	for r in range(w-1):
		pixel_white = 0
		start = 0
		end = 0
		for c in range(h-1):
			if img[r,c] == 255:
				pixel_white += 1
			if img[r, c] == 0 and img[r,c+1] == 255:
				start = c
			if img[r, c] == 255 and img[r,c+1] == 0:
				end = c
		
		if pixel_white > thshold*h:
			# print("pixel_white ", pixel_white , thshold*h)
			img[r,0:w] = 255
		else:
			img[r,0:w] = 0
	return img

def findMinMaxRow(v_img):
	aleft, aright = 0, 0
	list_col = []
	w, h = v_img.shape[0], v_img.shape[1]
	for r in range(w-1):
		pixel_white = 0
		for c in range(h-1):
			if v_img[r,c] == 255:
				pixel_white += 1
		if pixel_white > 20:
			list_col.append(r)
	aleft, aright = min(list_col), max(list_col)
	return aleft, aright

def getLines(img):
	lines = []
	w, h = img.shape[0], img.shape[1]
	for r in range(w-1):
		pixel_white = 0
		startx, starty, endx, endy = 0,0,0,0
		for c in range(h-1):
			if img[r,c] == 0 and img[r,c+1] == 255:
				startx = c
				starty = r
			if img[r,c] == 255 and img[r,c+1] == 0:
				endx = c
				endy = r
			if img[r,c] == 255:
				pixel_white += 1
		if pixel_white > 20:
			lines.append(Line(startx,starty,endx,endy))
			#print(Line(startx,starty,endx,endy).toArray())
	return lines

def findTable(arr, min_size = 10):
	table = defaultdict(list)
	
	for i,b in enumerate(arr):
		if b[2] < b[3]/2 or b[2] < min_size or b[3] <min_size :
			continue
		table[str(b[1])].append(b)
	#print(table)
	table = [i[1] for i in table.items()]# if len(i[1]) > 1]
	#print(([len(x) for x in table]))
	num_cols = max([len(x) for x in table])
	
	#print("num_cols:",num_cols)
	table = [i for i in table if len(i) == num_cols]
	#print("table rows=", len(table))
	#print("table cols=",num_cols)
	return table

def getTable(src_img, y_start=0, min_w=20, min_h=20):
	if y_start != 0:
		src_img = src_img[y_start:,:]
	if len(src_img.shape) == 2:
		gray_img = src_img
	elif len(src_img.shape) ==3:
		gray_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

	thresh_img = cv2.adaptiveThreshold(~gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, -3)
	h_img = thresh_img.copy()
	v_img = thresh_img.copy()
	scale = 5

	h_size = int(h_img.shape[1]/scale)
	h_structure = cv2.getStructuringElement(cv2.MORPH_RECT,(h_size,1))
	
	h_erode_img = cv2.erode(h_img,h_structure,1)
	h_dilate_img = cv2.dilate(h_erode_img,h_structure,1)
	
	v_size = int(v_img.shape[0] / scale)
	v_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_size))
	
	v_erode_img = cv2.erode(v_img, v_structure, 1)
	v_dilate_img = cv2.dilate(v_erode_img, v_structure, 1)
	
	
	# aleft, aright = findMinMaxRow(v_dilate_img.T)
	# aleft2, aright2 = findMinMaxRow(h_dilate_img)
	h_dilate_img = reDrawLine(h_dilate_img)
	
	v_dilate_img = reDrawLine(v_dilate_img.T).T
	
	# cv2.imwrite('v_dilate_img.jpg',v_dilate_img)
  
	# v_dilate_img.T[aleft,aleft2:aright2] = 255
	# v_dilate_img.T[aright,aleft2:aright2] = 255
	
	# edges = cv2.Canny(h_dilate_img,50,150,apertureSize = 3) 
	# h, w = edges.shape[:2]
	# #print(len(edges))

	mask_img = h_dilate_img + v_dilate_img
	# joints_img = cv2.bitwise_and(h_dilate_img, v_dilate_img)
	# #mask_img = 255 - mask_img
	# #mask_img = unsharp_mask(mask_img)
	# convolution_kernel = np.array(
	# 							[[0, 1, 0], 
	# 							[1, 2, 1], 
	# 							[0, 1, 0]]
	# 							)
	v_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	
	mask_img = cv2.dilate(mask_img, v_structure, 1)
	mask_img = cv2.erode(mask_img, v_structure, 1)
	contours, _ = cv2.findContours(mask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	
	(contours, boundingBoxes) = cont.sort_contours(contours, method="left-to-right")
	(contours, boundingBoxes) = cont.sort_contours(contours, method="top-to-bottom")
	# for i,cell in enumerate(boundingBoxes):
	# 	src_img = cv2.rectangle(src_img, (cell[0], cell[1]), (cell[0] + cell[2], cell[1] + cell[3]), (0,255,0), 2)
	# cv2.imwrite("t2.jpg", src_img)
	# print("contours " , boundingBoxes)
	table = findTable([cv2.boundingRect(x) for x in contours])
	index_start = -1 
	table_result = []
	h , w = mask_img.shape[:2]
	for i,row in enumerate(table):
		cell = row[0]
		if cell[3]*cell[2] > 0.8 * w *h:
			continue
		if cell[3] < min_h : 
			if i == index_start +1 :
				table_result.append(row)
			else:
				index_start = i
		else:
			table_result.append(row)
					
	return table_result

def drawTable (image , table, status_table ):
	for i,row in enumerate(table):
		for cell in row:
			image = cv2.rectangle(image, (cell[0], cell[1]), (cell[0] + cell[2], cell[1] + cell[3]), (0,255,0), 2)
			image = cv2.putText(image, f'{status_table[i]}', (cell[0] + cell[2]//2, cell[1] + cell[3]//2), cv2.FONT_HERSHEY_SIMPLEX, 
				   1, (0 , 0 , 255), 2, cv2.LINE_AA)
	return image

def drawTable2(image , table ):
	for i,row in enumerate(table):
		for cell in row:
			image = cv2.rectangle(image, (cell[0], cell[1]), (cell[0] + cell[2], cell[1] + cell[3]), (0,255,0), 2)
	return image

if __name__ == '__main__':
	
	folder_save = "results"
	if not os.path.exists(folder_save):
		os.mkdir(folder_save)
	imagePaths = sorted(list(paths.list_images("LK_image_from_pdf")))
	for imagePath in imagePaths:
		print("path " ,  imagePath)
		imagePath = "LK_image_from_pdf/LK-57TC-02_page0.jpeg"
		basename = os.path.basename(imagePath)
		image = cv2.imread(imagePath)
		scale = image.shape[0]/img.shape[0]
		box = [733, 381 ,888, 1086 ]
		img = image[box[1]:box[3], box[0]:box[2]]
		table = getTable(img)
		
		output = drawTable2(img, table)
		path_out =os.path.join(folder_save ,basename)
		cv2.imwrite(path_out, img)
		exit()