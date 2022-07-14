import cv2
import pytesseract

class TessRecognizer():
	def __init__(self):
		self.tess_config_eng = r'-l eng --psm 7 --oem 1'
		self.tess_config_digits = r'-l digits --psm 10 --oem 1'
		self.tess_config_japanese = r'-l japanese --psm 7 --oem 1 -c tessedit_char_blacklist=ABCDEFGHIJKLMNOPQRSTUVWXYZ'
		self.tess_config_hiragana = r'-l japanese --psm 7 --oem 1 -c tessedit_char_whitelist=あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわゐゑを'
		self.tess_config_japanese_digits = r'-l japanese --psm 7 --oem 1 -c tessedit_char_whitelist=1234567890-'

	def getEnglishString(self, image):
		text = pytesseract.image_to_string(image, config=self.tess_config_eng)
		return text

	def getDigitString(self, image):
		text = pytesseract.image_to_string(image, config=self.tess_config_digits)
		return text

	def getJapaneseString(self, image):
		text = pytesseract.image_to_string(image, config=self.tess_config_japanese)
		return text

	# def getJapaneseString
