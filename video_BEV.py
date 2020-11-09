"""
USAGE:
python video_BEV.py
python video_BEV.py -r True
python video_BEV.py --input pedestrians.mp4 -r True
python video_BEV.py --input pedestrians.mp4 --output output.avi

"""
from pyimagesearch import social_distancing_config as config
from pyimagesearch.detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import argparse
import logging
import imutils
import cv2
import os

logging.basicConfig(level=logging.DEBUG)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="pedestrians.mp4",
    help="path to (optional) input video file; path to webcam if not specified")
ap.add_argument("-o", "--output", type=str, default="",
    help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,
    help="whether or not output frame should be displayed")
#ap.add_argument("-r", "--region", type=bool, default=True,
#ap.add_argument("-r", "--region", type=bool, default=False,
#    help="whether or not BEV region should be specified by user")
ap.add_argument("-r", "--region", action="store_true",
    help="whether or not BEV region should be specified by user")
args = vars(ap.parse_args())

logging.debug(f"\nargs['region'] = {args['region']}")

labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

if config.USE_GPU:
    print("[INFO] setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

print("[INFO] accessing video stream...")
#vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
vs = cv2.VideoCapture(args["input"] if args["input"] else 3)
writer = None
writer_bev = None

is_1st_frame = True
corners=[]
count=0
borderpx = 15
dmenubarpx = 30

# loop over the frames from the video stream
while True:
    (grabbed, frame) = vs.read()

    if not grabbed:
        break

    #frame = imutils.resize(frame, width=700)
    # Resizing is for easy viewing, not because we have to. Try commenting out the
    # above line and you'll see the code run as well as when there was `imutil.resize()`

    ####################################################
    # We use the 1st frame to
    #   i) Draw rectangular region
    #  ii) Caculate homography matrix
    # iii)
    if is_1st_frame:
        region_color = (152,17,169)
        corner_thickness = 7
        edge_thickness = 2
        if not args["region"]:
            corners = config.CORNERS
        else:
            tmp = frame.copy()
            def draw_corners(event, x, y, flags, param):
                global corners, count
                if event == cv2.EVENT_LBUTTONUP:
                    cv2.circle(tmp, (x,y), radius=0, color=region_color, thickness=corner_thickness)
                    #corners.append([x,y])
                    corners.append((x,y))
                    count += 1

            win1 = "draw"
            cv2.namedWindow(win1)
            cv2.setMouseCallback(win1, draw_corners)

            while True:
                cv2.imshow(win1, tmp)
                if cv2.waitKey(33) & 0xFF == ord('q'):
                    print("You've pressed on `q`. This forces the program to end w/o doing anything useful.")
                    exit()
                elif count >= 4:
                    break
            #for i in range(len(corners)):
            #    cv2.line(tmp, corners[i], corners[(i+1)%4], color=region_color, thickness=edge_thickness)
            cv2.imshow(win1, tmp)
            cv2.destroyAllWindows()

        rect_from = np.array(corners, dtype=np.float32)
        print("corners = {}".format(corners))
        #w_to = 600
        #h_to = 2*w_to
        #h_to = w_to//2
        #h_to = w_to
        # ftown3.mp4
        w_to = 300
        h_to = int(2.2*w_to)
        rect_to = np.array([
            [0, 0],
            [w_to-1, 0],
            [w_to-1, h_to-1],
            [0, h_to-1],
            ], dtype=np.float32)

        H, _ = cv2.findHomography(rect_from, rect_to, cv2.RANSAC, 5.0)
        #H = cv2.getPerspectiveTransform(rect_src, rect_dst)
    ####################################################


    # bev stands for "Bird's Eye View"
    #logging.debug(f"type(cv2.warpPerspective(tmp, H, (w_to, h_to))) = {type(cv2.warpPerspective(tmp, H, (w_to, h_to)))}")
    #logging.debug(f"cv2.warpPerspective(tmp, H, (w_to, h_to)).shape = {cv2.warpPerspective(tmp, H, (w_to, h_to)).shape}")
    #logging.debug(f"type(np.zeros((h_to, w_to), dtype=np.uint8)) = {type(np.zeros((h_to, w_to), dtype=np.uint8))}")
    #logging.debug(f"np.zeros((h_to, w_to), dtype=np.uint8).shape = {np.zeros((h_to, w_to), dtype=np.uint8).shape}")
    bev = cv2.warpPerspective(frame, H, (w_to, h_to))
    #bev = np.zeros((h_to, w_to, 3), dtype=np.uint8)

    results = detect_people(frame, net, ln,
        personIdx=LABELS.index("person"))

    violate = set()

    if len(results) >= 2:
        centroids = np.array([r[2] for r in results])
        pieds = np.array([r[3] for r in results])
        bird_pieds = cv2.perspectiveTransform(pieds[np.newaxis, ...].astype(np.float32), H)[0]
        #D = dist.cdist(centroids, centroids, metric="euclidean")
        D = dist.cdist(bird_pieds, bird_pieds, metric="euclidean")
        for i in range(0, D.shape[0]):
            for j in range(i + 1, D.shape[1]):
                if D[i, j] < config.MIN_DISTANCE:
                    violate.add(i)
                    violate.add(j)

    #for (i, (prob, bbox, centroid, pied)) in enumerate(results):
    for (i, (prob, bbox, centroid, _)) in enumerate(results):
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        pX, pY = bird_pieds[i].astype(np.int)
        color = (0, 255, 0)

        if i in violate:
            color = (0, 0, 255)

        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        #cv2.circle(frame, (cX, cY), 5, color, 1)
        cv2.circle(bev, (pX, pY), radius=0, color=color, thickness=20)
        cv2.circle(bev, (pX, pY), radius=config.MIN_DISTANCE//2, color=color, thickness=3)

    #text = "Social Distancing Violations: {}".format(len(violate))
    text = "Violations: {}".format(len(violate))
    cv2.putText(frame, text, (10, frame.shape[0] - 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

    for i in range(len(corners)):
        cv2.circle(frame, corners[i], radius=0, color=region_color, thickness=corner_thickness)
        cv2.line(frame, corners[i], corners[(i+1)%4], color=region_color, thickness=edge_thickness)

    combined = np.zeros((max(frame.shape[0], bev.shape[0]),
                         frame.shape[1] + bev.shape[1],
                         3),
                         dtype=np.uint8)
    combined[:frame.shape[0], :frame.shape[1], :] = frame
    combined[:bev.shape[0], frame.shape[1]:, :] = bev

    if args["display"] > 0:
        """
        win2 = "bbox"
        cv2.namedWindow(win2)
        #cv2.moveWindow(win2, frame.shape[1] + 2*borderpx, dmenubarpx)
        cv2.moveWindow(win2, 0, dmenubarpx)
        cv2.imshow(win2, frame)

        win3 = "bev"
        cv2.namedWindow(win3)
        cv2.moveWindow(win3, frame.shape[1] + 2*borderpx, dmenubarpx)
        cv2.imshow(win3, bev)
        """

        win4 = "combined"
        cv2.namedWindow(win4)
        #cv2.moveWindow(win4, 0, dmenubarpx + max(frame.shape[0], bev.shape[0]) + 2*borderpx)
        #cv2.moveWindow(win4, 0, 0)
        cv2.imshow(win4, combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # if an output video file path has been supplied and the video
    # writer has not been initialized, do so now
    if args["output"] != "" and writer is None:
        # initialize our video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 25,
            (frame.shape[1], frame.shape[0]), True)
        head, ext = args["output"].split('.')
        bev_file = '.'.join([head + "-BEV", ext])
        writer_bev = cv2.VideoWriter(bev_file, fourcc, 25,
            (combined.shape[1], combined.shape[0]), True)

    # if the video writer is not None, write the frame to the output
    # video file
    if writer is not None:
        writer.write(frame)
    if writer_bev is not None:
        writer_bev.write(combined)
    is_1st_frame = False
