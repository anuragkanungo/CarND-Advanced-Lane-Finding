import glob
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from PIL import Image
from moviepy.editor import VideoFileClip

def save_image(img, name, directory="output_images", convert_bgr=True):
    if convert_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    Image.fromarray(img).save("{}/{}".format(directory, name))

def save_threshold_image(img, color_binary, combined_binary, name, directory="output_images"):
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8,3))
    f.tight_layout()
    ax1.set_title('Original Image')
    ax1.imshow(img)
    ax2.set_title('Stacked Thresholds')
    ax2.imshow(color_binary)
    ax3.set_title('Combined Thresholds')
    ax3.imshow(combined_binary, cmap='gray')
    f.savefig(os.path.join(os.getcwd(), directory, name))

def save_warped_image(img, warped, name, directory="output_images"):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(6,3))
    f.tight_layout()
    ax1.set_title('Combined Binary')
    ax1.imshow(img, cmap='gray')
    ax2.set_title('Warped Image')
    ax2.imshow(warped, cmap='gray')
    f.savefig(os.path.join(os.getcwd(), directory, name))

def save_hist(image, name, directory="output_images"):
    f = plt.figure()
    plt.plot(np.sum(image[image.shape[0]//2:,:], axis=0))
    f.savefig(os.path.join(os.getcwd(), directory, name))


def sliding_window(binary_warped, name=None, directory=None, save=False):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    if save:
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        f = plt.figure()
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        f.savefig(os.path.join(os.getcwd(), directory, name))
    return left_fit, right_fit


def selection_window(binary_warped, left_fit, right_fit, name=None, directory=None, save=False):
    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
    left_fit[1]*nonzeroy + left_fit[2] + margin))) 

    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
    right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    if save:
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                      ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                      ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        f = plt.figure()
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        f.savefig(os.path.join(os.getcwd(), directory, name))
    
    return left_fitx, right_fitx


def curvature(binary_warped, left_fit, right_fit):
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    leftx = left_fitx[::-1]  # Reverse to match top-to-bottom in y
    rightx = right_fitx[::-1] # Reverse to match top-to-bottom in yield

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    y_eval = np.max(ploty)

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    return left_curverad, right_curverad, ploty


def draw_lane_area(warped, undist, Minv, left_fitx, right_fitx, ploty, left_curverad, right_curverad, name=None, directory=None, save=False):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (warped.shape[1], warped.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    
    text = "Left Curvature: {:.2f} m".format(left_curverad)
    cv2.putText(result, text, (50,50), 1, 2, (255,255,255), 4)
    text = "Right Curvature: {:.2f} m".format(right_curverad)
    cv2.putText(result, text, (50,100), 1, 2, (255,255,255), 4)

    if save:
        f = plt.figure()
        result = img = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        plt.imshow(result)
        f.savefig(os.path.join(os.getcwd(), directory, name))

    return result


def calibrate_camera(dir_path):
    images = glob.glob("{}/calibration*.jpg".format(dir_path))
    nx, ny = 9, 6
    
    objpoints = []
    imgpoints = []

    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1, 2)

    for file in images:
        img = cv2.imread(file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            save_image(img, "corners_{}".format(file.split("/")[-1]), directory=dir_path)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx, dist

def perspective_transform(image):
    src = np.float32([[580, 460], [700, 460], [1040, 680], [260, 680]])
    dst = np.float32([[200, 0], [1040, 0], [1040,720], [200,720]])

    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))
    # Return the resulting image and matrix
    return warped, M, Minv


def threshold(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    # Convert to HLS color space and separate the L and S channels
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    #print(hls.shape)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Stack each channel
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return color_binary, combined_binary


def test_images(dir_path, output_path, mtx, dist):
    images = glob.glob("{}/*.jpg".format(dir_path))
    for file in images:
        img = cv2.imread(file)
        undistort = cv2.undistort(img, mtx, dist, None, mtx)
        save_image(undistort, "undisoted_{}".format(file.split("/")[-1]), directory=output_path)
        color_binary, combined_binary = threshold(undistort)
        save_threshold_image(img, color_binary, combined_binary, "thresh_{}".format(file.split("/")[-1]), directory=output_path)
        warped, M, Minv = perspective_transform(combined_binary)
        save_warped_image(combined_binary, warped, "warped_{}".format(file.split("/")[-1]), directory=output_path)
        save_hist(warped, "hist_{}".format(file.split("/")[-1]), directory=output_path)
        left_fit, right_fit = sliding_window(warped, "sliding_window_{}".format(file.split("/")[-1]), directory=output_path, save=True)
        left_fitx, right_fitx = selection_window(warped, left_fit, right_fit, "selection_window_{}".format(file.split("/")[-1]), directory=output_path, save=True)
        l, r, ploty = curvature(warped, left_fit, right_fit)
        result = draw_lane_area(warped, undistort, Minv, left_fitx, right_fitx, ploty, l, r, "final_{}".format(file.split("/")[-1]), directory=output_path, save=True)


def find_lanes(img, mtx, dist):
    undistort = cv2.undistort(img, mtx, dist, None, mtx)
    color_binary, combined_binary = threshold(undistort)
    warped, M, Minv = perspective_transform(combined_binary)
    left_fit, right_fit = sliding_window(warped)
    left_fitx, right_fitx = selection_window(warped, left_fit, right_fit)
    l, r, ploty = curvature(warped, left_fit, right_fit)
    result = draw_lane_area(warped, undistort, Minv, left_fitx, right_fitx, ploty, l, r)
    return result

def process_video(mtx, dist, input_path, output_path):
    def process_image(image):
        return find_lanes(image, mtx, dist)
    clip = VideoFileClip(input_path)
    processed_clip = clip.fl_image(process_image)
    processed_clip.write_videofile(output_path, audio=False)


def detect_lane_lines_pipeline():
    mtx, dist = calibrate_camera("camera_cal")
    print('mtx: ', mtx)
    print('dist: ', dist)
    test_images("test_images", "output_images", mtx, dist)
    process_video(mtx, dist, input_path="project_video.mp4", output_path="output_project_video.mp4")
    process_video(mtx, dist, input_path="challenge_video.mp4", output_path="output_challenge_video.mp4")
    process_video(mtx, dist, input_path="harder_challenge_video.mp4", output_path="output_harder_challenge_video.mp4")



##############################################################
# Run the Lane Line Detection Pipeline
##############################################################

detect_lane_lines_pipeline()