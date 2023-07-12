import cv2 as cv # Import OpenCV library as cv
import numpy as np # Import NumPy library as np

# Load the image as grey scale
image = cv.imread("input.png", cv.IMREAD_GRAYSCALE)


# Apply a Gaussian blur to image with a 5x5 kernel and default border 
blur = cv.GaussianBlur(image, (5, 5), cv.BORDER_DEFAULT)

# Apply a binary inverse threshold to the blurred image 
ret, thresh = cv.threshold(blur, 200, 255, cv.THRESH_BINARY_INV)

# Find the contours in the thresholded image using 
contours, _ = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

# Create a blank image with same theshold size
blank = np.zeros(thresh.shape, dtype='uint8')

# Draw the contours on the blank image in blue color with a thickness of 1 pixel
cv.drawContours(blank, contours, -1, (255, 0, 0), 1)

# Add a border of 200 pixels to all sides of the blank image with a constant border type and value of 0
blank = cv.copyMakeBorder(blank, 200, 200, 200, 200, cv.BORDER_CONSTANT, None, value=0)

# Create a circle stamp with a shape of 100x100 and data type uint8
circle = np.zeros((100, 100), dtype="uint8")

# Draw a circle on the circle stamp with center at (50,50)
circle = cv.circle(circle, (50, 50), 60, 10, 2)

# Get the height and width of the blank image
img_h, img_w = blank.shape
# Create a copy of the blank image for output
output = blank.copy()

# Iterate through the pixels of the blank image
for y in range(0, img_h):
    for x in range(0, img_w):
        # If the pixel value is greater than 200
        if blank[y, x] > 200:
            # Check if the shape of the region around the current pixel is not equal to (100,100)
            if output[y - 50:y + 50, x - 50:x + 50].shape != (100, 100):
                break # If it is not equal to (100,100), break out of the inner loop
            # Add the circle stamp to the region around the current pixel in the output image
            output[y - 50:y + 50, x - 50:x + 50] += circle[0:100, 0:100]

# Write the output image to a file with the specified file path
cv.imwrite("CenteringCircle.png", output)
