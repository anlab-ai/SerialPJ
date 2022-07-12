
import argparse
import os
import math
from turtle import position
import cv2
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import numpy as np
import recognize_check_sheet
from common import ErrorCode
import imutils
import constant
import csv
# import SerialDetection
# import tess_recognitor
outputFolder = 'output'
outputFile = 'results.csv'
checkSheetReader = recognize_check_sheet.CheckSheetReader()

def readResults(filePath):
	errCode = ErrorCode.SUCCESS
	resultFile = open(filePath)
	csvreader = csv.reader(resultFile)
	results = {}
	for row in csvreader:
		listRawData = row[1:]
		results[row[0]] = listRawData
	return errCode, results


def readImage(filePath):
	print("filePath = ", filePath)
	image = cv2.imread(filePath)
	image = cv2.resize(image,[int(image.shape[1] * constant.A4_FORM_HEIGHT / image.shape[0]),constant.A4_FORM_HEIGHT])
	errorCode, info = checkSheetReader.RecognizeForm(image, os.path.basename(filePath))
	if errorCode == ErrorCode.SUCCESS:
		with open(outputFile, 'a') as resultsFile:
			resultsFile.write(info + '\n')

if __name__=="__main__":
	parser = argparse.ArgumentParser(description='test')
	if os.path.exists(outputFile):
		os.remove(outputFile)

	example = 2
	if example == 1:
		parser.add_argument('--image', type=str, default='input/test.jpg', help='path image')
		args = parser.parse_args()
		if not os.path.exists(args.image):
					print("Invalid parameters provided")
		readImage(args.image)
	elif example == 2:
		parser.add_argument('--input', type=str, default='input', help='path dir')
		args = parser.parse_args()
		if not os.path.exists(args.input):
			print("Invalid parameters provided")
		files = os.listdir(args.input)
		for file in files:
			readImage(args.input + "/" + file)

		#statistic
		errCode, expectedResults = readResults("expected_result_70_images.csv")
		print(f'len(expectedResults) = {len(expectedResults)}')
		errCode, results = readResults("results.csv")
		print(f'len(results) = {len(results)}')
		# print(f'len title = {len(expectedResults["File"])}')
		# print(f'titles = {expectedResults["File"]}')
		correctResults = [0] * len(expectedResults['File'])
		missingResults = 0
		alertResult = [0] * len(expectedResults['File'])
		for file in results:
			# print(f'file = {file}')
			if file in expectedResults:
				# print(f'len(results[{file}] = {len(results[file])}')
				for i in range(len(results[file])):
					if i in range(len(expectedResults[file])):
						if expectedResults[file][i] == results[file][i] :
							correctResults[i] += 1
						elif len(results[file][i]) > 0 and results[file][i][0] == '[' and results[file][i][-1] == ']':
								print(f'alert file : {file}')
								alertResult[i] += 1
						else:
							if expectedResults["File"][i] == 'SerialNo':
								print(f'file : {file}')
								print(f'results[file][i] = {results[file][i]}')
			else:
					missingResults += 1
		for i in range(len(expectedResults['File'])):
			print(f'{correctResults[i]}')
		for i in range(len(expectedResults['File'])):
			print(f'Alert: {alertResult[i]}')

		print(f'Total result: {len(results)} , Missing: {missingResults}')		


	exit()

