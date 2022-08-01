import io
import os
import cv2
from google.cloud import vision

class TextRecognizer: 
    def __init__(self):
        self.key = "cloudvision-357810-fdfa9e0b5e39.json"


    def detect_text_path(self, path):
        with io.open(path, 'rb') as image_file:
            content = image_file.read()

        image = vision.Image(content=content)
        return self.detect_text_image(image)

    def detect_text_cv2_image(self, cv2_image):
      success, encoded_image = cv2.imencode('.png', cv2_image)
      content2 = encoded_image.tobytes()
      image = vision.Image(content=content2)
      return self.detect_text_image(image)
      
    def detect_text_image(self, image):
      labels = []
      positions = []
      """Detects text in the file."""
      os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.key
      client = vision.ImageAnnotatorClient()
      response = client.text_detection(image=image)
      texts = response.text_annotations
      sentence = response.full_text_annotation.text

      print(f'Texts: {texts}')
      for text in texts:
          #self.labels.append('\n"{}"'.format(text.description))
          labels.append('{}'.format(text.description))

          vertices = (['({},{})'.format(vertex.x, vertex.y)
                        for vertex in text.bounding_poly.vertices])

          #print('bounds: {}'.format(','.join(vertices)))
          positions.append('bounds: {}'.format(','.join(vertices)))
  
      if response.error.message:
          raise Exception(
              '{}\nFor more info on error messages, check: '
              'https://cloud.google.com/apis/design/errors'.format(
                  response.error.message))

      return labels, sentence, positions

if __name__ == "__main__":
  textRecognizer = TextRecognizer()
  labels, sentence, positions = textRecognizer.detect_text_path("input/MFG No.032200710 LK-F57VC-04_pumpname.jpg")
  print(f"labels = {labels}")
