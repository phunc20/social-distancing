# base path to YOLO directory
MODEL_PATH = "yolo-coco"

# initialize minimum probability to filter weak detections along with
# the threshold when applying non-maxima suppression
MIN_CONF = 0.3
NMS_THRESH = 0.3

# boolean indicating if NVIDIA CUDA GPU should be used
USE_GPU = False

# define the minimum safe distance (in pixels) that two people can be
# from each other
#MIN_DISTANCE = 50
#MIN_DISTANCE = 90
MIN_DISTANCE = 110

#CORNERS = [(326, 48), (672, 90), (587, 256), (92, 152)]
#CORNERS = [(293, 63), (643, 102), (508, 322), (9, 190)]
CORNERS = [(0, 96), (171, 107), (318, 405), (92, 432)]
