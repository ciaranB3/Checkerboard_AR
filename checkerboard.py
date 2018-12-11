"""
checkerboard.py 

This script detects a chessboard pattern on an image coming from the 
	computer's built-in camera. 
!! COMMENTED OUT, just used for testing!! It utilises undistort parameters calculated in "CalibScript.py" to 
	!! undistort the incoming frames. 
It then projects a jpeg image onto the chessboard 

Author: 	Ciaran Bannon

"""

import numpy as np
import cv2

if __name__ == "__main__":

	# size of the checkerboard
	WIDTH = 6
	HEIGHT = 9

	# load undistortion parameters and create cameramtx
	mtx =  np.array([
		(642.0903131074885550, 0.0,                  305.3215785839721548), 
		(0.0,                  643.4730274808072181, 262.0910459527405010), 
		(0.0,                  0.0,                  1.0)])
	dist = np.array([(0.2009489901541785251, -1.601153569956497824, 
		0.004644091806523309440, -0.001154489791305557400, 5.856777192576116065)])
	w, h = 640, 480
	newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(640,480))

	# load the image to overlay
	dsp = cv2.imread("Dragon.jpg")

	# this array stores the pixel corners of the image 
	imgSrc = np.array([
		[0, 0],
		[dsp.shape[1]-1, 0],
		[dsp.shape[1] - 1, dsp.shape[0] - 1],
		[0, dsp.shape[0] - 1]], dtype = "float32")

	cap = cv2.VideoCapture(0)

	while(True):

		#capture a frame
		ret, img = cap.read()

		# undistort
		# img = cv2.undistort(img, mtx, dist, None, newcameramtx)

		# convert to greyscale
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		# Find the chess board corners
		ret, corners = cv2.findChessboardCorners(gray, (HEIGHT,WIDTH),None)

		if ret == True:
			# identify the edge corners
			rect = np.zeros((4, 2), dtype = "float32")
			yPoints = corners[:,0,1]
			xPoints = corners[:,0,0]
			rect[0] = [xPoints[0],yPoints[0]]
			rect[1] = [xPoints[8],yPoints[8]]
			rect[2] = [xPoints[-1],yPoints[-1]]
			rect[3] = [xPoints[-9],yPoints[-9]]
			 
			# calculate the perspective transform matrix and warp
			M = cv2.getPerspectiveTransform(imgSrc, rect)
			warp = cv2.warpPerspective(dsp, M, (w, h))

			# identify the black regions of the image and 
			# replace with the video capture
			warp[warp==0.0] = img[warp==0.0]
			img = warp

		cv2.imshow('img',img)      
		    
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	        
	# release everything
	cap.release()
	cv2.destroyAllWindows()