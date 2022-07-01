import os
import cv2
import utilitiesProcessImage
import DigitsDetection
import SerialDetection
import tess_recognitor
from common import ErrorCode
import constant
import Contours
from enum import Enum
import numpy as np

import csv



folder_save = "results"

class OCRMode(Enum):
	DIGIT = 0
	HAND_WRITTING_DIGIT_STYPE_JP = 1
	ENGLISH = 2
	HAND_WRITTING_ENGLISH = 3
	JAPANESE = 4
	HAND_WRITTING_JAPANESE = 5
	HAND_WRITTING_SERIAL_STYPE_JP = 6
	COUNT = 7

class CheckSheetReader():
	def __init__(self):
		self.model = SerialDetection.SerialDetection()
		self.hwDigitsStyleJpModel = DigitsDetection.DigitDetection()
		self.tessRecognitor = tess_recognitor.TessRecognitor()
		self.readPositionCsvFile("./position_forms/lk.csv")
		checkMaterialDefaultImg = cv2.imread("./template_check_material/check_material_template.jpg")
		gray = cv2.cvtColor(checkMaterialDefaultImg, cv2.COLOR_BGR2GRAY)
		binaryImg = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 7)
		# morphKernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(2, 2))
		# self.maskCheckMaterial = cv2.morphologyEx(binaryImg, cv2.MORPH_CLOSE, morphKernel2)
		kernel = np.ones((11,11), np.uint8)
		self.maskCheckMaterial = cv2.dilate(binaryImg, kernel, iterations=1)
		
	def readPositionCsvFile(self, filePath):
		positionFile = open(filePath)
		csvreader = csv.reader(positionFile)
		self.position_infos = {}
		for row in csvreader:
			self.position_infos[row[0]] = [int(row[1]),int(row[2]),int(row[3]),int(row[4])]

	def RecognizeForm(self, image, imgName):
		errCode = ErrorCode.SUCCESS
		
		if utilitiesProcessImage.startDebug == True:
			cv2.imshow("image", image)
		# utilitiesProcessImage.startDebug = True
		pumpName = ''
		mfgNo = ''
		motorLotNo = ''
		maker = ''
		construction = ''
		side = ''
		electricType = ''
		powerValue = ''
		dynamicViscosity = ''
		pumpValue = ''
		vValue = ''
		hzValue = ''
		minValue = ''
		serial_number = ''
		flangePHeadMaterial = ''
		valveMaterial = ''
		vGuideMaterial = ''
		gasketConfirmation = ''
		vMaterial = ''
		oRingMaterial = ''
		errCode, imgPumpName, pumpName = self.readPumpName(image)
		errCode, imgMFGNo, mfgNo = self.readMFGNo(image)
		errCode, imgMotorLotNo , motorLotNo = self.getString(image,self.position_infos[constant.TAG_MOTOR_LOT_NO], OCRMode.HAND_WRITTING_SERIAL_STYPE_JP)
		errCode, imgMaker, maker = self.getString(image, self.position_infos[constant.TAG_MAKER], OCRMode.JAPANESE)
		errCode, imgConstruction, construction = self.getString(image, self.position_infos[constant.TAG_CONSTRUCTION], OCRMode.JAPANESE)
		side , electricType = self.model.checkSelection(image)
		# print(f'side , electricType = {(side , electricType)}')
		errCode, imgPowerValue, powerValue = self.getString(image, self.position_infos[constant.TAG_POWER_VALUE], OCRMode.DIGIT)
		errCode, imgDynamicViscosity, dynamicViscosity = self.getString(image, self.position_infos[constant.TAG_DYNAMIC_VISCOSITY], OCRMode.DIGIT)
		errCode, imgPumpValue, pumpValue = self.getString(image, self.position_infos[constant.TAG_PUMP_VALUE], OCRMode.ENGLISH)
		errCode, imgVValue , vValue = self.getString(image,self.position_infos[constant.TAG_V_VALUE], OCRMode.DIGIT)
		errCode, imgHzValue , hzValue = self.getString(image,self.position_infos[constant.TAG_HZ_VALUE], OCRMode.DIGIT)
		errCode, imgMinValue , minValue = self.getString(image,self.position_infos[constant.TAG_MIN_VALUE], OCRMode.HAND_WRITTING_SERIAL_STYPE_JP)
		# errCode, img_serial , serial_number = self.getString(image,self.position_infos[constant.TAG_SERIAL_NO], OCRMode.HAND_WRITTING_SERIAL_STYPE_JP)
		img_serial , serial_number = self.model.getSerialForm(image)
		errCode, imgCheckMaterial, checkMaterialSelections = self.detectSelectionCheckMaterial(image)
		flangePHeadMaterial = str(checkMaterialSelections[0])
		valveMaterial = str(checkMaterialSelections[1])
		vGuideMaterial = str(checkMaterialSelections[2])
		gasketConfirmation = str(checkMaterialSelections[3])
		vMaterial = str(checkMaterialSelections[4])
		oRingMaterial = str(checkMaterialSelections[5])
		
		infoStr = f'{imgName},{pumpName.strip().replace(",", "")},{mfgNo.strip().replace(",", "")},{motorLotNo.strip().replace(",", "")}\
,{maker.strip().replace(",", "")},{construction.strip().replace(",", "")},{side},{electricType}\
,{powerValue.strip().replace(",", "")},{dynamicViscosity.strip().replace(",", "")},{pumpValue.strip().replace(",", "")}\
,{vValue.strip().replace(",", "")},{hzValue.strip().replace(",", "")},{minValue.strip().replace(",", "")},{serial_number.strip().replace(",", "")}\
,{flangePHeadMaterial},{valveMaterial},{vGuideMaterial}\
,{gasketConfirmation},{vMaterial},{oRingMaterial}'.strip().replace('"', '')

		#save to view output image (test)
		# utilitiesProcessImage.startDebug = True
		if utilitiesProcessImage.startDebug == True:
			# path_out =os.path.join(folder_save , f'{imgName}')
			# cv2.imwrite(path_out, imgPumpName)
			# path_out =os.path.join(folder_save , f'{imgName}')
			# cv2.imwrite(path_out, imgMFGNo)
			# path_out =os.path.join(folder_save , f'{lotNo}_{imgName}')
			# cv2.imwrite(path_out, imgLotNo)
			# path_out =os.path.join(folder_save , f'{imgName}')
			# cv2.imwrite(path_out, imgPumpName)
			# path_out =os.path.join(folder_save , f'{vValue}_{imgName}')
			# cv2.imwrite(path_out, imgVValue)
			# path_out =os.path.join(folder_save , f'{hzValue}_{imgName}')
			# cv2.imwrite(path_out, imgHzValue)
			# path_out =os.path.join(folder_save , f'{imgName}')
			# cv2.imwrite(path_out, imgMinValue)
			# path_out =os.path.join(folder_save , f'{imgName}')
			# cv2.imwrite(path_out, imgCheckMaterial)
			# path_out =os.path.join(folder_save , f'{serial_number}_{imgName}')
			# cv2.imwrite(path_out, img_serial)
			print(f'infoStr: {infoStr}')
			cv2.waitKey(0)
			utilitiesProcessImage.startDebug = False
		return errCode, infoStr

	def detectSelectionCheckMaterial(self, image):
		errCode = ErrorCode.SUCCESS
		box = self.position_infos[constant.TAG_FLANGE_P_HEAD_MATERIAL]
		# utilitiesProcessImage.startDebug = True
		if utilitiesProcessImage.startDebug:
				print(f'box = {box}')
		outputImg = image[box[1]:box[3],box[0]:box[2]]
		errCode, binImg = utilitiesProcessImage.convertBinaryImage(outputImg)
		errCode, y_line = utilitiesProcessImage.findBorderHorizontalLine(binImg, 0.6,isTopLine=True)
		y_line = max(0, y_line)
		# oriBinImage = binImg.copy()
		errCode, binImg = utilitiesProcessImage.removeHorizontalLineTable(binImg, 0.6, 5)
		# binImg = cv2.bitwise_and(binImg, binImg1)
		# if utilitiesProcessImage.startDebug == True:
		# 	cv2.imshow("binImg_bitwise_and", binImg)
		errCode, binImg = utilitiesProcessImage.filterBackgroundByColor(outputImg, binImg, 200)
		
		h,w = self.maskCheckMaterial.shape
		if utilitiesProcessImage.startDebug:
			print(f'h,w = {h},{w}')
			print(f'y_line = {y_line}')
			cv2.waitKey()
		binImg = binImg[y_line:y_line+h,:w]
		outputImg = outputImg[y_line:y_line+h,0:w]
		mask = cv2.bitwise_not(self.maskCheckMaterial)
		if binImg.shape[0] < h or binImg.shape[1] < w:
			mask = mask[:binImg.shape[0], :binImg.shape[1]]
		if utilitiesProcessImage.startDebug:
			cv2.imshow("mask", mask)
		binImg = cv2.bitwise_and(binImg,mask)
		if utilitiesProcessImage.startDebug:
			cv2.imshow("binImg", binImg)
			cv2.waitKey()
	
		selections = [-1] * 6
		for i in range(6):
			itemImg = binImg[int(i*binImg.shape[0]/6):int((i+1)*binImg.shape[0]/6)]
			numberOptions = 0
			if i % 2 == 0:
				numberOptions = 6
			else:
				numberOptions = 3
			maxCountNonZero = 10
			maxHeight = 0
			for j in range(numberOptions):
				optionImg = itemImg[:,int(j*itemImg.shape[1]/numberOptions):int((j+1)*itemImg.shape[1]/numberOptions)]
				err, optionImg = utilitiesProcessImage.removeVerticalLineTable(optionImg, 0.6, 5)	
				if utilitiesProcessImage.startDebug:
					if i == 1:
						cv2.imshow(f"optionImg{i*6 + j}", optionImg)
				err,box = utilitiesProcessImage.getContentArea(optionImg, 2)
				if err == ErrorCode.SUCCESS:
					if box[3] > constant.CHAR_HEIGHT and box[3] > maxHeight:
						maxHeight = box[3]
						selections[i] = j
					elif maxHeight == 0:
						countNonZero = cv2.countNonZero(optionImg)
						if countNonZero > maxCountNonZero:
							maxCountNonZero = countNonZero
							selections[i] = j

		# errCode, selectionBoxs = utilitiesProcessImage.findSelectionByContour(binImg)
		# for selectionBox in selectionBoxs:
		# 	if utilitiesProcessImage.startDebug == True:
		# 		cv2.rectangle(outputImg, selectionBox, (0,0,255), 2)
		# 	itemIndex = int(((selectionBox[1] + selectionBox[3]/2) * 6) / (box[3] - box[1]))
		# 	selectIndex = int(((selectionBox[0] + selectionBox[2] / 2) * 6) / (box[2] - box[0]))
		# 	if utilitiesProcessImage.startDebug == True:
		# 		print(f'itemIndex = {itemIndex} , selectIndex = {selectIndex}')
		# 	# if itemIndex not in selections:
		# 	selections[itemIndex] = selectIndex

		if utilitiesProcessImage.startDebug:
			utilitiesProcessImage.startDebug = False
			print(f'selections = {selections}')
			cv2.imshow("outputImg", outputImg)
			
		return errCode, binImg, selections

	def readPumpName(self, image):
		errCode = ErrorCode.SUCCESS
		box = self.position_infos[constant.TAG_PUMP_NAME]
		outputImg = image[box[1]:box[3],box[0]:box[2]]
		# utilitiesProcessImage.startDebug = True
		if utilitiesProcessImage.startDebug:
			cv2.imshow("readPumpName_outputImg",outputImg)
		gray = cv2.cvtColor(outputImg, cv2.COLOR_BGR2GRAY)
		binImg = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 7)
		if utilitiesProcessImage.startDebug:
			cv2.imshow("readPumpName_binImg",binImg)
		oriBinImg = binImg.copy()
		errCode, binImg = utilitiesProcessImage.removeHorizontalLineTable(binImg, 0.6, 9)
		errCode, binImg = utilitiesProcessImage.filterBackgroundByColor(outputImg, binImg, 200)
		
		errCode, box_info = utilitiesProcessImage.findMainArea(binImg,2)
		padding = int(box_info[3]*0.5)

		# ocrImg = binImg[max(box_info[1] - padding,0):min(box_info[1]+box_info[3], outputImg.shape[0]), max(box_info[0], 0):min(box_info[0]+box_info[2], outputImg.shape[1])]
		# cv2.copyMakeBorder(outputImage, padding, padding, padding*3, padding*3, cv2.BORDER_CONSTANT, None, value = 0)
		
		ocrImg = binImg[max(box_info[1] - padding,0):min(box_info[1]+box_info[3] + padding, outputImg.shape[0]), max(box_info[0] - padding, 0):min(box_info[0]+box_info[2] + padding, outputImg.shape[1])]
		
		ocrImg = cv2.bitwise_not(ocrImg)
		# utilitiesProcessImage.startDebug = True
		if utilitiesProcessImage.startDebug:
			cv2.imshow("readPumpName_ocrImage",ocrImg)
		h,w = ocrImg.shape
		errCode, outputImg, outputText = self.getString(ocrImg,[0,0,w,h], OCRMode.ENGLISH)
		outputText = outputText.replace("~","-").replace('=','-').replace('--','-').replace('$','S').replace('?','7').replace('O','0').replace('-S','-5').replace('-FS','-F5').replace('-L','-1').replace(' ','')
		if utilitiesProcessImage.startDebug:
			utilitiesProcessImage.startDebug = False
			print(f"outputText: {outputText}")
			cv2.waitKey()
		return errCode, outputImg, outputText

	def readMFGNo(self, image):
		errCode = ErrorCode.SUCCESS
		box = self.position_infos[constant.TAG_MFG_NO]
		outputImg = image[box[1]:box[3],box[0]:box[2]]
		# utilitiesProcessImage.startDebug = True
		if utilitiesProcessImage.startDebug == True:
			cv2.imshow("readMFGNo_outputImg",outputImg)
		gray = cv2.cvtColor(outputImg, cv2.COLOR_BGR2GRAY)
		binImg = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 7)
		if utilitiesProcessImage.startDebug == True:
			cv2.imshow("readMFGNo_binImg",binImg)
		oriBinImg = binImg.copy()
		errCode, binImg = utilitiesProcessImage.removeHorizontalLineTable(binImg, 0.6, 7)
		errCode, binImg = utilitiesProcessImage.filterBackgroundByColor(outputImg, binImg, 200)
		
		# ocrImg = binImg

		errCode, box_info = utilitiesProcessImage.findMainArea(binImg,2)
		padding = int(box_info[3]*0.5)

		ocrImg = oriBinImg[max(box_info[1] - padding,0):min(box_info[1]+box_info[3] + padding, outputImg.shape[0]), max(box_info[0] - 3*padding, 0):min(box_info[0]+box_info[2] + 3*padding, outputImg.shape[1])]
		
		# ocrImg = binImg[max(box_info[1],0):min(box_info[1]+box_info[3], outputImg.shape[0]), max(box_info[0], 0):min(box_info[0]+box_info[2], outputImg.shape[1])]
		# ocrImg = cv2.copyMakeBorder(ocrImg, padding, padding, padding, padding, cv2.BORDER_CONSTANT, None, value = 0)
		
		ocrImg = cv2.bitwise_not(ocrImg)
		if utilitiesProcessImage.startDebug == True:
			cv2.imshow("readMFGNo_ocrImage",ocrImg)
		h,w = ocrImg.shape
		errCode, outputImg, outputText = self.getString(ocrImg,[0,0,w,h], OCRMode.ENGLISH)
		return errCode, outputImg, outputText.replace(' ', '')

	def getString(self, image, box, mode):
		errCode = ErrorCode.SUCCESS
		outputStr = ""
		outputImg = None
		if mode == OCRMode.HAND_WRITTING_SERIAL_STYPE_JP:
			img_serial = image[box[1]:box[3],box[0]:box[2]]
			listChar = Contours.splitCharFromForm(img_serial)
			est = 3
			h, w = img_serial.shape[:2]
			for i, box_char in enumerate(listChar):
				xmin = max(0 , box_char[0][0]-est)
				ymin = max(0 , box_char[0][1] -est)
				xmax = min(w , box_char[1][0] + est)
				ymax = min(h , box_char[1][1] + est)
				im_char = img_serial[ymin:ymax, xmin:xmax]
				idx_cls, bestclass, bestconf = self.model.predict(im_char , False )
				outputStr += bestclass
			outputImg = img_serial

		elif mode == OCRMode.HAND_WRITTING_DIGIT_STYPE_JP:
			outputImg , listChar = Contours.splitCharFromForm(image, box)
			for i, box in enumerate(listChar):
				xmin = box[0][0]
				ymin = box[0][1]
				xmax = box[1][0]
				ymax = box[1][1]
				im_char = outputImg[ymin:ymax, xmin:xmax]
				idx_cls, bestclass, bestconf = self.hwDigitsStyleJpModel.predict(im_char , pose= i)
				outputStr += bestclass

		elif mode == OCRMode.ENGLISH:
			outputImg = image[box[1]:box[3],box[0]:box[2]]
			outputStr = self.tessRecognitor.getEnglishString(outputImg)
		elif mode == OCRMode.DIGIT:
			outputImg = image[box[1]:box[3],box[0]:box[2]]
			outputStr = self.tessRecognitor.getDigitString(outputImg)
		elif mode == OCRMode.JAPANESE:
			outputImg = image[box[1]:box[3],box[0]:box[2]]
			outputStr = self.tessRecognitor.getJapaneseString(outputImg)
		
		if outputStr is None:
			outputStr = ''
		return errCode, outputImg , outputStr

