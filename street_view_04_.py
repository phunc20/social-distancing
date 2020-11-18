"""
USAGE:
python street_view_04.py
python street_view_04.py -i street-views/street-view-04.jpg

"""
import cv2
import logging
import numpy as np
from pyimagesearch.transform import order_points

logging.basicConfig(level=logging.DEBUG)

from pyimagesearch import social_distancing_config as config
from pyimagesearch.detection import detect_people
from scipy.spatial import distance as dist
import argparse
import imutils
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="street-views/street-view-04.jpg",
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

corners=[] 
count=0
#src = cv2.imread("street-views/street-view-04.jpg")
tmp = resized.copy()

def draw_corners(event, x, y, flags, param):
    global corners, count
    # If event is Left Button Click then store the coordinate in the lists, corners and 
    if event == cv2.EVENT_LBUTTONUP:
        cv2.circle(tmp, (x,y), radius=0, color=(0,200,0), thickness=7)
        corners.append([x,y])
        count += 1

win_draw = "draw"
cv2.namedWindow(win_draw)
cv2.setMouseCallback(win_draw, draw_corners)

while True:
    # Note that in order to update the circular points that we draw on win_draw
    # we must put cv2.imshow() inside the while loop.
    cv2.imshow(win_draw, tmp)
    # 27 is Esc key (in case you want to use it)
    if cv2.waitKey(33) & 0xFF == ord('q') or count >= 4:
        break
cv2.imshow(win_draw, tmp)

#cv2.destroyAllWindows()
print("corners = {}".format(corners))
w_dst = 300
h_dst = 2*w_dst
h_src, w_src = resized.shape[:2]

#rect_src = order_points(corners)
print(f"order_points(corners) = {order_points(corners)}")
#rect_src = corners
rect_src = np.array(corners, dtype=np.float32)
#rect_src = corners.astype(np.float32)
#rect_dst = order_points(corners)
rect_dst = np.array([
    [0, 0],
    [w_dst-1, 0],
    [w_dst-1, h_dst-1],
    [0, h_dst-1],
    ], dtype=np.float32)

#H, _ = cv2.findHomography(rect_src, rect_dst, cv2.RANSAC, 5.0)
H = cv2.getPerspectiveTransform(rect_src, rect_dst)
warped1 = cv2.warpPerspective(resized, H, (w_dst, h_dst))

#cv2.imshow("warped1", warped1)
#while True:
#    k = cv2.waitKey(33)
#    if k == ord('q'):
#        break


violate = set()

# ensure there are *at least* two people detected (required in
# order to compute our pairwise distance maps)
if len(results) >= 2:
    # extract all centroids from the results and compute the
    # Euclidean distances between all pairs of the centroids
    centroids = np.array([r[2] for r in results])
    pieds = np.array([r[3] for r in results])
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

#######################################
## Draw centroids on Bird's Eye View ##
#######################################
logging.debug(f"\ncentroids =\n{centroids}")
bird_centroids = cv2.perspectiveTransform(centroids[np.newaxis, ...].astype(np.float32), H)
bird_pieds = cv2.perspectiveTransform(pieds[np.newaxis, ...].astype(np.float32), H)
logging.debug(f"\nbird_centroids =\n{bird_centroids}")
logging.debug(f"\ncorners =\n{corners}")
logging.debug(f"\ncv2.perspectiveTransform(rect_src[np.newaxis, ...].astype(np.float32), H) =\n{cv2.perspectiveTransform(rect_src[np.newaxis, ...].astype(np.float32), H)}")

#for c in np.squeeze(bird_centroids, axis=0):
for c in np.squeeze(bird_pieds, axis=0):
    cv2.circle(warped1, tuple(c.astype(np.int)), radius=0, color=(0,200,0), thickness=30)

cv2.imshow("Centroids in BEV", warped1)
while True:
    k = cv2.waitKey(33)
    if k == ord('q'):
        break


# loop over the results
for (i, (prob, bbox, centroid, pied)) in enumerate(results):
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

h, w = original.shape[:2]
rect_dst = np.array([
    [0, 249],
    [w-1, 249],
    [w-1, h-1],
    [0, h-1]])



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
