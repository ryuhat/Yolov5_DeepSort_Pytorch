import cv2

# Load the image containing the fish
image = cv2.imread("J:\My Drive\yolov8_tracking\data/tuna/tuna1.png")

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create a SIFT object
sift = cv2.xfeatures2d.SIFT_create()

# Detect keypoints and descriptors in the image
keypoints, descriptors = sift.detectAndCompute(gray, None)

# Sort the keypoints by their response (strength)
keypoints = sorted(keypoints, key=lambda x: -x.response)

# Extract the two keypoints with the highest response as the head and tail keypoints
head_keypoint = keypoints[0]
tail_keypoint = keypoints[1]

# Draw circles around the head and tail keypoints in the image
cv2.circle(image, (int(head_keypoint.pt[0]), int(head_keypoint.pt[1])), int(head_keypoint.size/2), (0, 0, 255), 2)
cv2.circle(image, (int(tail_keypoint.pt[0]), int(tail_keypoint.pt[1])), int(tail_keypoint.size/2), (0, 255, 0), 2)

# Display the image with the detected keypoints
cv2.imshow("Fish keypoints", image)
cv2.waitKey(0)
