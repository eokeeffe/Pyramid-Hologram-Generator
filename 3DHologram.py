import cv2
import numpy as np
import os,sys

def makeHologram(original,scale=0.5,scaleR=4,distance=0):
    '''
        Create 3D hologram from image (must have equal dimensions)
    '''
    
    height = int((scale*original.shape[0]))
    width = int((scale*original.shape[1]))
    
    image = cv2.resize(original, (width, height), interpolation = cv2.INTER_CUBIC)
    
    up = image.copy()
    down = rotate_bound(image.copy(),180)
    right = rotate_bound(image.copy(), 90)
    left = rotate_bound(image.copy(), 270)
    
    hologram = np.zeros([max(image.shape)*scaleR+distance,max(image.shape)*scaleR+distance,3], image.dtype)
    
    center_x = (hologram.shape[0])/2
    center_y = (hologram.shape[1])/2
    
    vert_x = (up.shape[0])/2
    vert_y = (up.shape[1])/2
    hologram[0:up.shape[0], center_x-vert_x+distance:center_x+vert_x+distance] = up
    hologram[ hologram.shape[1]-down.shape[1]:hologram.shape[1] , center_x-vert_x+distance:center_x+vert_x+distance] = down
    hori_x = (right.shape[0])/2
    hori_y = (right.shape[1])/2
    hologram[ center_x-hori_x : center_x-hori_x+right.shape[0] , hologram.shape[1]-right.shape[0]+distance : hologram.shape[1]+distance] = right
    hologram[ center_x-hori_x : center_x-hori_x+left.shape[0] , 0+distance : left.shape[0]+distance ] = left
    
    #cv2.imshow("up",up)
    #cv2.imshow("down",down)
    #cv2.imshow("left",left)
    #cv2.imshow("right",right)
    #cv2.imshow("hologram",hologram)
    #cv2.waitKey()
    return hologram

def process_video(video):
    cap = cv2.VideoCapture(video)

    # Define the codec and create VideoWriter object
    fourcc = cv2.cv.CV_FOURCC(*'XVID')
    holo = None
    ret = False
    while(not ret):
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 640), interpolation = cv2.INTER_CUBIC)
            holo = makeHologram(frame)
    out = cv2.VideoWriter('hologram.avi',fourcc, 30.0, (holo.shape[0],holo.shape[1]))
    total_frames = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
    count = 0
    print "Processing %d frames"%(total_frames)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 640), interpolation = cv2.INTER_CUBIC)
            holo = makeHologram(frame)
            out.write(holo)
            count += 1
            print "Total:%d of %d"%(count,total_frames)
        if(count>=total_frames-1):
            break
    
    # Release everything if job is finished
    cap.release()
    out.release()
    return

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))
    
if __name__ == '__main__' :
    orig = cv2.imread(sys.argv[1])
    holo = makeHologram(orig,scale=1.0)
    process_video("/home/evan/Videos/test.avi")
    #cv2.imwrite("hologram.png",holo)
