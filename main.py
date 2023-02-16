import cv2
import numpy as np
import matplotlib.pyplot as plt

from math_utils import RT_from_3D

# =========================================================== #
# ======================= GLOBAL VARS ======================= #
# =========================================================== #

# Camera intrinsics
K = np.array([[3.214738872901308696e+03,	0.000000000000000000e+00,	1.988303498905410834e+03],
                       [0, 3.213956395514384440e+03, 9.741750736623685043e+02],
                       [0, 0, 1]])

# =========================================================== #
# ======================== FUNCTIONS ======================== #
# =========================================================== #

def get_corners(im, chessboard_size):
    
    # convert to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    # get the corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    
    # Remove redundant dimensions
    corners = np.squeeze(corners)

    return corners
    
def draw_corners(im, corners):
    
    # returned image
    ret = np.array(im)
    
    # Change to integer for pixel drawing
    corners_int = corners.astype(int)
    
    for corner in corners_int:
        center = (corner[0],corner[1])
        cv2.circle(ret, center, 5, (0,0,255), 25)
    
    return ret

def draw_cube(im, base_pts, ceiling_pts):
    
    # returned image
    ret = np.array(im)
    
    # draw base
    cv2.drawContours(ret, [base_pts], -1, (0,0,255), 10)
    
    # draw top
    cv2.drawContours(ret, [ceiling_pts], -1, (0, 0, 255), 10)
    
    # draw pillars
    for i,j in zip(base_pts, ceiling_pts):
        cv2.line(ret, i, j, (0,0,255), 10)
    
    return ret


# =========================================================== #
# ======================== MAIN LOOP ======================== #
# =========================================================== #


if __name__ == '__main__':
    
    # Chessboard params
    chessboard_size = (9, 6)
    square_size = 26 # mm
    
    # ==================== WORLD POINTS ==================== #
    
    # Define the points in the real world
    M = np.zeros((np.prod(chessboard_size), 3), dtype=np.float32)
    M[:, :2] = np.indices(chessboard_size).T.reshape(-1, 2)
    M *= square_size
    M_ = np.delete(M, 2, 1) # World coordinates on checkerboard (No Z axis)
    M_homog = np.hstack((M, np.ones((M.shape[0], 1)))) # Homogenous coordinates
    
    # Cube world points
    indices = [1, 7, 52, 46] # Corner indices (hardcoded)
    base = M_homog[indices].T
    ceiling = np.array(base)
    ceiling[2,:] = square_size * -1.5
    
    # ===================== VIDEO PARAMS ===================== #
    
    # Define the video capture
    cap = cv2.VideoCapture('videos_in/checkerboard.mp4')
    
    # Create output video writer
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('videos_out/output_video.mp4', fourcc, fps, (width, height))
    
    count = 0 # for counting iterations while the program is running
    
    while True:
        
        # Capture frame-by-frame
        ret, frame = cap.read()

        print(count, ret)
        count+=1
    
        if not ret:
            break
    
        # get corners in image plane
        m = get_corners(frame, chessboard_size)
    
        # Rotation and translation matrix
        R, T = RT_from_3D(m[indices], M_[indices], K)
        # Projection matrix
        P = np.concatenate((R, T), axis = 1)
        
        # Base points in camera plane
        base_pxls = K @ P @ base
        base_pxls2 = base_pxls[:-1,:] / base_pxls[-1,:]
        base_pxls2 = base_pxls2.T.astype(int)
    
        # Celing points in camera plane
        ceiling_pxls = K @ P @ ceiling
        ceiling_pxls2 = ceiling_pxls[:-1,:] / ceiling_pxls[-1,:]
        ceiling_pxls2 = ceiling_pxls2.T.astype(int)
        
        result = draw_cube(frame, base_pxls2, ceiling_pxls2)
        out.write(result)
        
cap.release()
out.release()
