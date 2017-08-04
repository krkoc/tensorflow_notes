import cv2
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import random as rn
#test

 
# load the image and show it
cols=28
rows=28
import numpy as np
for i in range (1,10000): #how many files
	pattern = Image.new("RGB", (28, 28), "white")
	size = width, height = pattern.size
	draw = ImageDraw.Draw(pattern)
	fontSize=rn.randint(16,19)  #range of font size
	font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", fontSize)
	digit=rn.randint(0,9)
	positionX=rn.randint(0,1)  #range of position
	positionY=rn.randint(0,1) #range of position

	draw.text((positionX,positionY), str(digit), (0, 0, 0, 255),font=font)
	image=np.array(pattern)

	#cv2.imshow("original", image)
	rotationAngle=rn.randint(-9,9) #range of rotation
	M = cv2.getRotationMatrix2D((cols/2,rows/2),rotationAngle,1)
	white_image = np.zeros(28, np.uint8)
	image = cv2.warpAffine(image, M, (cols, rows),
	flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

	stretchHorizontal=rn.uniform(1,1.1)  #range of stretch
	stretchVertical=rn.uniform(1,1.1) # range of stretch

	image = cv2.resize(image,None,fx=stretchHorizontal, fy=stretchVertical, interpolation = cv2.INTER_CUBIC)
	image = image[0:28,0:28]
	cv2.imshow("modified",image)
	cv2.imwrite( "/home/peter/tensorflow_scripts/nminst/"+str(digit)+"_"+str(i)+".png", image );
	
