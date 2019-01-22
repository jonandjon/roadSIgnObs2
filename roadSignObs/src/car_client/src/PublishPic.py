#!/usr/bin/env python

from cv_bridge import CvBridge
import cv2
import numpy as np
import rospy
import time
import sys
import random
#-# from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img # for roadSignObs
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Int32
from PIL import Image
import csv ###
import skimage #?#

pause = 4 # sekunden
img_rows, img_cols = 32, 32  # input image dimensions

'''
Liest Bilder aus dem Testbildverzeichnis und schickt sie zur Bilderkennungssoftware
'''
class publishpic:
	def __init__(self):
        	self.cv_bridge = CvBridge()
       		# publish Pictures to prediction
        	self.publisherPicture = rospy.Publisher("/camera/output/webcam/compressed_img_msgs", 
										CompressedImage,queue_size=1)
		 
	
		 #--------------------------------------------------------------------------------------
	
	# liest Bilddateien -----------------------------------
	def readSigns(self, rootpath="./TestImages"):
		size=[img_rows,img_cols]
		npImages = [] # images
		prefix = rootpath + '/' # Pfad zum Bilder-Verzeichnis
		gtFile = open(prefix + 'GT-final_test.test.csv') # annotations file
		gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
		gtReader.next() # skip header
		# loop over all images in current annotations file
		for row in gtReader:
			dateiname=prefix + row[0]
			ppmImage=Image.open(dateiname)
			## npNormImg=np.resize(ppmImage,size) geht nicht gut
			ppmNormImage=ppmImage.resize(size)  ## Standardgroesse herstellen
			#~# ppmNormImage=skimage.color.rgbcie2rgb(ppmNormImage)
			#+# nparray = np.array(jpgNormImage)
			npImage=img_to_array(ppmNormImage, data_format = "channels_last")
			# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_core/py_basic_ops/py_basic_ops.html
			b, g, r = cv2.split(npImage)
			npImage=cv2.merge((r,g,b))
			npImages.append(npImage)
		gtFile.close()
		return npImages 
		
		

	# liefert zufaeliges Bild aus Bildersammlung	
	def getRandomImages(self, images):
		zufallsindex=random.randint(0, len(images)-1)
		# print("Zufallsindex of Images: %d" %  zufallsindex) 
		image=images[zufallsindex]
		print ('Image: '+format(zufallsindex, '05d')+'.ppm')
		return image 
	
	## veroeffentlicht Daten
	def publishPicture(self, image, verbose=0):
        	# convert 
		# npImage=img_to_array(image, data_format = "channels_last")
        	compressed_imgmsg = self.cv_bridge.cv2_to_compressed_imgmsg(image)
        	# publish data
        	#+# self.publisherPicture.publish(compressed_imgmsg)
		self.publisherPicture.publish(compressed_imgmsg.header, compressed_imgmsg.format, compressed_imgmsg.data)
		#?# self.publisherPicture.publish(compressed_imgmsg.format, compressed_imgmsg.data,  compressed_imgmsg.data)
        	if verbose:
			rospy.loginfo(compressed_imgmsg.header.seq)
			rospy.loginfo(compressed_imgmsg.format)
				
		 			
				
def main():
	verbose = 0  # use 1 for debug
	# register node
	rospy.init_node('publishpic', anonymous=False)
	# init CameraPseudo
	pict = publishpic()
	# Hole die Bilder zum Testen ab
	images=pict.readSigns() #rootpath="./TestImages"
	# start publishing data
	print("Anzahl der zu bewertenden Bilder: ", len(images))

	# Scleife kann mit ^C beendet werden
	while not rospy.is_shutdown():
		im=random.randint(0, len(images)-1)
		image=images[im]
		## image.show()#t
		time.sleep(pause) 
		pict.publishPicture(image, verbose)
		print ( 'Image: '+format(im, '05d')+'.ppm')
		time.sleep(pause) 

if __name__ == '__main__':
    main()
