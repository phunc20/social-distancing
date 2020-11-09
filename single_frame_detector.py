# USAGE
# python single_frame_detector.py --i WFH-awards.jpg
# python single_frame_detector.py --input conference-AI.jpg

# import the necessary packages
from pyimagesearch import social_distancing_config as config
from pyimagesearch.detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
    help="path to (optional) input video file")
#ap.add_argument("-o", "--output", type=str, default="",
#    help="path to (optional) output video file")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# check if we are going to use GPU
if config.USE_GPU:
    # set CUDA as the preferable backend and target
    print("[INFO] setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


original = cv2.imread(args["input"])
resized = imutils.resize(original, width=700)
graffiti = resized.copy()
results = detect_people(graffiti, net, ln, personIdx=LABELS.index("person"))

violate = set()

# ensure there are *at least* two people detected (required in
# order to compute our pairwise distance maps)
if len(results) >= 2:
    # extract all centroids from the results and compute the
    # Euclidean distances between all pairs of the centroids
    centroids = np.array([r[2] for r in results])
    D = dist.cdist(centroids, centroids, metric="euclidean")
    # D returns a (symmetric) matrix composed of distances btw centroids

    # loop over the upper triangular of the distance matrix
    for i in range(0, D.shape[0]):
        for j in range(i + 1, D.shape[1]):
            # check to see if the distance between any two
            # centroid pairs is less than the configured number
            # of pixels
            if D[i, j] < config.MIN_DISTANCE:
                # update our violation set with the indexes of
                # the centroid pairs
                violate.add(i)
                violate.add(j)

# loop over the results
for (i, (prob, bbox, centroid, _)) in enumerate(results):
    # extract the bounding box and centroid coordinates, then
    # initialize the color of the annotation
    (startX, startY, endX, endY) = bbox
    (cX, cY) = centroid
    color = (0, 255, 0) # GREEN

    # if the index pair exists within the violation set, then
    # update the color to RED
    if i in violate:
        color = (0, 0, 255) # RED

    # draw (1) a bounding box around the person and (2) the
    # centroid coordinates of the person,
    cv2.rectangle(graffiti, (startX, startY), (endX, endY), color, 2)
    cv2.circle(graffiti, (cX, cY), 5, color, 1)

# draw the total number of social distancing violations on the
# output frame
text = "Social Distancing Violations: {}".format(len(violate))
cv2.putText(graffiti, text, (10, graffiti.shape[0] - 25),
    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

win1 = "Original"
cv2.namedWindow(win1)
# y_tl is the y coord. of the top left corner of the image to be shown in win1
y_tl = 30
cv2.moveWindow(win1, 0, y_tl)
#cv2.imshow(win1, original)
cv2.imshow(win1, resized)

win2 = "Graffiti"
cv2.namedWindow(win2)
cv2.moveWindow(win2, resized.shape[1], y_tl)
cv2.imshow(win2, graffiti)

while True:
    k = cv2.waitKey(33)
    if k == ord('q'):
        break
