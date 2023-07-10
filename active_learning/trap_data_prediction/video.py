#!/usr/bin/env python
# coding: utf-8
""""
Author        : Aditya Jain
Date started  : July 18, 2022
About         : builds a video given the images and their classfication, localization, tracking information
"""

import cv2
import os
import time
import pandas as pd
import argparse

def prev_track_centre(annot_data, img_name, track_id):
    """returns centre given a track id and image, if available"""
    
    if track_id == 'NA':
        return [None, None]

    img_points = annot_data[annot_data['image']==img_name]
    
    for i in range(len(img_points)):
        if img_points.iloc[i,1]==track_id:
            return [img_points.iloc[i,6], img_points.iloc[i,7]]
     
    return [None, None]
    

def make_video(args):
	"""main function for making the video"""
	
	# User-defined variables
	data_dir     = args.data_dir
	image_folder = args.image_folder
	framerate    = args.frame_rate
	region       = args.region
	scale_fac    = args.scale_factor
	image_dir    = data_dir + image_folder + '/'
	track_file   = data_dir + 'track_localize_classify_annotation-' + image_folder + '.csv'
	video_name   = region + '_' + image_folder + '_' + 'track_localize_classify_binary_finegrained.mp4'
	
	data_images = os.listdir(image_dir)
	data_annot  = pd.read_csv(track_file)

	# Output video settings
	test_img    = cv2.imread(image_dir + data_images[0])
	height, width, layers = test_img.shape

	vid_out     = cv2.VideoWriter(data_dir + video_name,
                              cv2.VideoWriter_fourcc(*'XVID'), 
                              framerate, 
                              (int(width*scale_fac),int(height*scale_fac)))

	# Iterate over images
	i = 0
	prev_image_name = ''
	start           = time.time()
	img_count       = 0

	while i<len(data_annot):
		image_name = data_annot.loc[i, 'image']
		image = cv2.imread(image_dir + image_name)
		img_count += 1
    
		while i<len(data_annot) and data_annot.loc[i, 'image']==image_name:
			if data_annot.loc[i, 'class']=='nonmoth':
				cv2.rectangle(image,
                      (data_annot.loc[i, 'bb_topleft_x'], data_annot.loc[i, 'bb_topleft_y']),
                      (data_annot.loc[i, 'bb_botright_x'], data_annot.loc[i, 'bb_botright_y']),
                      (255,0,0),
                       3)
				cv2.putText(image, 'nonmoth: '+ str(data_annot.loc[i, 'confidence']) + '%', 
                    (data_annot.loc[i, 'bb_topleft_x'], data_annot.loc[i, 'bb_topleft_y']-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                    (255,0,0), 
                    2)        
				cv2.circle(image,
                   (data_annot.loc[i, 'bb_centre_x'], data_annot.loc[i, 'bb_centre_y']), 
                   4, 
                   (255,0,0), 
                   -1)
			else:
				cv2.rectangle(image,
                      (data_annot.loc[i, 'bb_topleft_x'], data_annot.loc[i, 'bb_topleft_y']),
                      (data_annot.loc[i, 'bb_botright_x'], data_annot.loc[i, 'bb_botright_y']),
                      (0,0,255),
                       3)
				cv2.putText(image, str(data_annot.loc[i, 'subclass']) + ': ' + str(data_annot.loc[i, 'confidence']) + '%', 
                    (data_annot.loc[i, 'bb_topleft_x'], data_annot.loc[i, 'bb_topleft_y']-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                    (0,0,255), 
                    2)
# 				cv2.putText(image, 'moth: '+ str(data_annot.loc[i, 'confidence']) + '%', 
#                     (data_annot.loc[i, 'bb_topleft_x'], data_annot.loc[i, 'bb_topleft_y']-10), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
#                     (0,0,255), 
#                     2)
				cv2.circle(image,
                   (data_annot.loc[i, 'bb_centre_x'], data_annot.loc[i, 'bb_centre_y']), 
                   4, 
                   (0,0,255), 
                   -1)
        
			# showing the previous track by drawing a line
			if prev_image_name:
				prev_centre = prev_track_centre(data_annot, prev_image_name, data_annot.loc[i, 'track_id'])
				if prev_centre[0]:
					cv2.line(image,
                         (prev_centre[0], prev_centre[1]),
                         (data_annot.loc[i, 'bb_centre_x'], data_annot.loc[i, 'bb_centre_y']),
                         (0,0,255), 
                          3)                
                
			i += 1    
    
		prev_image_name = image_name 
		image           = cv2.resize(image, None, fx= scale_fac, fy= scale_fac, interpolation= cv2.INTER_LINEAR)
# 		cv2.imwrite(data_dir + 'annotated_image.jpg',image)
		vid_out.write(image)
        
	vid_out.release()
	print('Time take to build video in minutes: ', (time.time()-start)/60)
	print('Total images: ', img_count)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--data_dir", help = "root directory containing the trap data", required=True)
	parser.add_argument("--image_folder", help = "date folder within root directory containing the images", required=True)
	parser.add_argument("--frame_rate", help = "frame rate of the resulting video", default=5, type=int)
	parser.add_argument("--scale_factor", help = "scale the raw image by this factor", default=0.5, type=float)
	parser.add_argument("--region", help = "name of the region", required=True)
	args   = parser.parse_args()
	make_video(args)




