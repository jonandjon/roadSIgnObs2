#!/usr/bin/env python
from cv_bridge import CvBridge
import cv2
import numpy as np
import rospy
#-# from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img # for roadSignObs
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool, Int32
from PIL import Image
# from threading import Thread
import threading
import time


PUBLISH_RATE = 5 # hz
USE_WEBCAM = True
img_rows, img_cols = 32, 32  # input image dimensions
PAUSE = 5 # sekunden

class PublishWebCam:
	def __init__(self):
		self.cv_bridge = CvBridge()
		# publish webcam
		self.publisher_webcam_comprs = rospy.Publisher("/camera/output/webcam/compressed_img_msgs",
                                                       CompressedImage,
                                                       queue_size=1)
													   
		# publish Frame
		self.publisher_fullcam_comprs = rospy.Publisher("/fullcamera/output/webcam/compressed_img_msgs",
                                                       CompressedImage,
                                                       queue_size=1)												

		if USE_WEBCAM:
			self.input_stream = cv2.VideoCapture(0)
			if not self.input_stream.isOpened():
				raise Exception('Camera stream did not open\n')

        rospy.loginfo("Publishing data...")
#--------------------------------------------------------------------------------------
	''' veroeffentlicht Daten '''
	def publish_data(self, verbose=0):
		rate = rospy.Rate(PUBLISH_RATE)
		while not rospy.is_shutdown():
			# Note:
			# reactivate for webcam image. Pay attention to required subscriber buffer size.
			# See README.md for further information
			if USE_WEBCAM:
				# Methode zum veroeffentlichen des Vollbildes 
				frame=self.publish_webcam(verbose)
				rate.sleep()
				# Methode zum Veroeffentlichen des sklalierten Bildes
				self.publish_camresize(frame)
	
			
	''' Sendet Vollbilder der Webcam fortlaufend '''
	def publish_webcam(self, verbose=0):
		if self.input_stream.isOpened():
			success, frame = self.input_stream.read()
			msg_frame = self.cv_bridge.cv2_to_compressed_imgmsg(frame)
			# -> Uebertragung der vollstaendigen-Camera-Bilder
			self.publisher_fullcam_comprs.publish(msg_frame.header, msg_frame.format, msg_frame.data)
			if verbose:
				rospy.loginfo(msg_frame.header.seq)
				rospy.loginfo(msg_frame.format)
		return frame
	
	''' Sendet skaliertes Bild der WebCam '''
	''' Methode alternativ als Thread https://www.python-kurs.eu/threads.php '''	
	def publish_camresize(self, frame):
		# print(time.time())
		# nicht jedes Bild, sondern entsprechend des Pausenzyklus
		if ((int(PUBLISH_RATE*time.time()) % (PAUSE*PUBLISH_RATE)) == 0):
			print("frame publish", int(time.time())) # Kontrollausgabe
			images=array_to_img(frame)		
			size=[img_rows, img_cols]
			jpgNormImage=images.resize(size)     # Standardgroesse herstellen
			npImage=img_to_array(jpgNormImage) ### als Numphy-Array
			compressed_imgmsg = self.cv_bridge.cv2_to_compressed_imgmsg(npImage)
			# -> Sendet skaliertes Bild 
			self.publisher_webcam_comprs.publish(compressed_imgmsg)
			
		
	
def main():
	verbose = 0  # use 1 for debug
	# register node
	rospy.init_node('PublishWebCam', anonymous=False)
	# Instanz der Klasse (Publisher)
	cam = PublishWebCam()
	# start publishing data
	cam.publish_data(verbose)

	try:
		rospy.spin()
	except rospy.ROSInterruptException:
        	pass
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main()
