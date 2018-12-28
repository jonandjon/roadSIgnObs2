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

pause = 5 # sekunden
img_rows, img_cols = 64, 64# input image dimensions

'''
Liest Bilder aus dem Testbildverzeichnis und schickt sie zur Bilderkennungssoftware
'''
class publishpic:
	def __init__(self):
        	self.cv_bridge = CvBridge()
       		# publish Pictures to prediction
        	self.publisherPicture = rospy.Publisher("/camera/output/webcam/compressed_img_msgs", CompressedImage,queue_size=10)

		 #--------------------------------------------------------------------------------------
	
	# liest Bilddateien -----------------------------------
	def readSigns(self, rootpath="./TestImages"):
	
		npImages = [] # images
		size=[img_rows,img_cols]
		# loop over all 42 classes
		# for c in range(0,subDirNo,1):
		prefix = rootpath + '/' # Pfad zum Bilder-Verzeichnis
		gtFile = open(prefix + 'GT-final_test.test.csv') # annotations file
		gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
		gtReader.next() # skip header
		#-# print("gtRead: ", gtReader)
		# loop over all images in current annotations file
		for row in gtReader:
			jpgImage=Image.open(prefix + row[0])
			jpgNormImage=jpgImage.resize(size)  ## Standardgroesse herstellen
			npImage=img_to_array(jpgNormImage)
			npImages.append(npImage)
		gtFile.close()
		npImages= np.array(npImages, dtype='float32') ##test
		print("Number of Images: %d" % len(npImages)) 
		return npImages
	
	## veroeffentlicht Daten
	def publishPicture(self, image, verbose=0):
        	#-# image = self.npImages[0]
        	# convert to images
        	compressed_imgmsg = self.cv_bridge.cv2_to_compressed_imgmsg(image)
        	# publish data
        	self.publisherPicture.publish(compressed_imgmsg)
			#-# self.publisherPicture.publish(compressed_imgmsg.header, compressed_imgmsg.format, compressed_imgmsg.data)
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
	try:

		# start publishing data
		print("Anzahl der zu bewertenden Bilder: ", len(images))
		for i in range (0, 22):
			im=random.randint(0, len(images)-1)
			pict.publishPicture(images[im], verbose)
			print (i, 'Image: '+format(im, '05d')+'.ppm')
			time.sleep(pause)
		#rospy.spin()
	except rospy.ROSInterruptException:
		pass


if __name__ == '__main__':
    main()
