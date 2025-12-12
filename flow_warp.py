import cv2
import numpy as np

# load image
prev = cv2.imread('./flow/0.png',0)
next = cv2.imread('./flow/2.png',0)

# change RGB to gray
# prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
# next_gray = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)
prev_gray = prev
next_gray = next

# calculate optical flow
flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

# calculate mat
w = int(prev.shape[1])
h = int(prev.shape[0])
y_coords, x_coords = np.mgrid[0:h, 0:w]
coords = np.float32(np.dstack([x_coords, y_coords]))
pixel_map = coords + flow/2
new_frame = cv2.remap(prev, pixel_map, None, cv2.INTER_LINEAR)
cv2.imshow("new",new_frame)


#fusion
cur = cv2.imread('./flow/1.png',0)
cv2.imshow("1",cur)
inter = (cur + new_frame)/2
inter=np.array(inter,dtype='uint8')
cv2.imshow("Image",inter)


cv2.imwrite('./flow/new.png', inter)
