import numpy as np
import cv2

# Shi-tomasi corner detection parameters
st_params = dict(maxCorners=30,
                 qualityLevel=0.2,
                 minDistance=2,
                 blockSize=7)

# Optical flow
"""Pattern of apparent motion of image objects between two consecutive
frames caused by movement of object or camera. It works on 2 assumptions:
1) Pixel densities of an object do not change between consecutive 
frames.
2) Neighbouring pixels have similar motion."""

# Lucas-Kanade method
"""Differential method for optical flow developed. It assumes that flow 
is essentially constant in a local neighbourhood of pixel under 
consideration, and solves basic optical flow equations for all pixels 
in that neighbourhood, by least squares criterion."""

# Lucas-Kanade optical flow parameters
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1))

# Capturing video
cap = cv2.VideoCapture('videos/car.mp4')

# Color for optical flow
color = (0, 255, 0)

# Reading capture and first frame
ret, first_frame = cap.read()

# Converting frame to grayscale
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Finding strongest corners
prev = cv2.goodFeaturesToTrack(prev_gray,
                               mask=None,
                               **st_params)

# Creating an image with same dimensions as frame for later drawing
# purposes
mask = np.zeros_like(first_frame)

while (cap.isOpened()):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculating optical flow by Lucas-Kanade
    next, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)

    # select good feature for previous position
    good_prev = prev[status == 1]

    # select good feature for next position
    good_next = next[status == 1]

    # drawing optical flow track
    for i, (new, old) in enumerate(zip(good_next, good_prev)):
        # Return coordinates for new point
        a, b = new.ravel()
        print(a,b)

        # Return coordinates for old point
        c, d = old.ravel()
        print(c,d)


        # Draw line between new and old position
        mask = cv2.line(mask, (a, b), (c, d), color, 2)

        # Draw filled circle
        frame = cv2.circle(frame,
                           (a, b),
                           2,
                           3,
                           -1)

    # Overlay optical flow on original frame
    output = cv2.add(frame, mask)

    # Update previous frame
    prev_gray = gray.copy()

    # Update previous good features
    prev = good_next.reshape(-1, 1, 2)

    # Open new window and display the output
    cv2.imshow("Optical Flow", output)

    # CLose the frame
    if cv2.waitKey(300) & 0xFF == ord("q"):
        break

# Release and Destroy
cap.release()
cv2.destroyAllWindows()



