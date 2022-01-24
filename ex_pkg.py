'''
test the pkgs
'''

import cv2
from skimage import io

##
# img = cv2.imread('examples/im1010.jpg')
# cv2.imshow('test', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

if __name__ == '__main__':
	img_io = io.imread('examples/im1010.jpg')
	print('img_io max min', img_io.max(), img_io.min()) # still 255 to 0