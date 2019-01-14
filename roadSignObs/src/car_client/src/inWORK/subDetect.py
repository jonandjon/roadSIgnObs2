#!/usr/bin/env python
'''
https://stackoverrun.com/de/q/4784088
https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html
is an another process using it:
https://askubuntu.com/questions/15433/unable-to-lock-the-administration-directory-var-lib-dpkg-is-another-process
 '''
#--- one ---
import cv2
import numpy as np 
from PIL import Image
# --- two ---
import os 
from itertools import izip 
from PIL import Image, ImageGrab 
 
# constants
IMAGE_SIZE = 200.0
MATCH_THRESHOLD = 3
CASCADEXML=[] # "haarcascade_obj.xml"  # lbpCascade.xml # haarcascade.xml (NOT WORK) # haarcascade_roundabout.xml
CASCADEXML.append("haarcascade_obj.xml") 	# <- TO WORK 0
CASCADEXML.append("lbpCascade.xml")		# <- TO WORK 1

class Object:
	def __init__(self):
		print ("hier ist der Objekt-Detektor")
	
	# --- one -----------
	def detectOne(self, analysebild="objDetect/street/mitKreisverkehr.png", objektbild="objDetect/objekt/kreisRotBig.jpg"):
		image 	= cv2.imread(analysebild) 
		template = cv2.imread(objektbild)
		
		result = cv2.matchTemplate(image,template,cv2.TM_CCOEFF_NORMED) 
		result = cv2.matchTemplate(image,template,cv2.TM_CCOEFF_NORMED) 
		print np.unravel_index(result.argmax(),result.shape) 
		#return streetImage, objImages
	# ---------------------------------------------------two--------https://stackoverrun.com/de/q/4784088----------------------
	def iter_rows(pil_image): 
		"""Yield tuple of pixels for each row in the image. 
		From: 
		http://stackoverflow.com/a/1625023/1198943 
		:param PIL.Image.Image pil_image: Image to read from. 
		:return: Yields rows. 
		:rtype: tuple 
		""" 
		iterator = izip(*(iter(pil_image.getdata()),) * pil_image.width) 
		for row in iterator: 
			yield row 
		
	def find_subimage(large_image, subimg_path): 
		"""Find subimg coords in large_image. Strip transparency for simplicity. 
		:param PIL.Image.Image large_image: Screen shot to search through. 
		:param str subimg_path: Path to subimage file. 
		:return: X and Y coordinates of top-left corner of subimage. 
		:rtype: tuple 
		""" 
		# Load subimage into memory. 
		with Image.open(subimg_path) as rgba, rgba.convert(mode='RGB') as subimg: 
			si_pixels = list(subimg.getdata()) 
			si_width = subimg.width 
			si_height = subimg.height 
			si_first_row = tuple(si_pixels[:si_width]) 
			si_first_row_set = set(si_first_row) # To speed up the search. 
			si_first_pixel = si_first_row[0] 

		# Look for first row in large_image, then crop and compare pixel arrays. 
		for y_pos, row in enumerate(iter_rows(large_image)): 
			if si_first_row_set - set(row): 
				continue # Some pixels not found. 
		for x_pos in range(large_image.width - si_width + 1): 
			if row[x_pos] != si_first_pixel: 
				continue # Pixel does not match. 
		if row[x_pos:x_pos + si_width] != si_first_row: 
			continue # First row does not match. 
				box = x_pos, y_pos, x_pos + si_width, y_pos + si_height 
		with large_image.crop(box) as cropped: 
			if list(cropped.getdata()) == si_pixels: 
		# We found our match! 
		return x_pos, y_pos 


	def find(subimg_path): 
		"""Take a screenshot and find the subimage within it. 
		:param str subimg_path: Path to subimage file. 
		""" 
		assert os.path.isfile(subimg_path) 
		# Take screenshot. 
		with ImageGrab.grab() as rgba, rgba.convert(mode='RGB') as screenshot: 
			print find_subimage(screenshot, subimg_path) 
	#-----------------------------------------------     