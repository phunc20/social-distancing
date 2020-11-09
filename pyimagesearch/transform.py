import cv2
import logging
import numpy as np

logging.basicConfig(level=logging.DEBUG)

def order_points(pts):
    """
    args:
    pts, ndarray of shape (4,2) (or a list of this shape)
        simply contains the 4 corners of a rectangle, in ANY order.

    res:
    rect, ndarray of shape (4,2)
        same points as those in pts, but ORDERED.
    """
    rect = np.zeros((4,2), dtype="float32")
    # What we want is order the points in pts as follows into rect:
    # rect[0] top-left
    # rect[1] top-right
    # rect[2] bottom-right
    # rect[3] bottom-left

    # Using the conventional coordinate system of computer vision,
    # we can say that top-left will have sum x+y smallest.
    # Similarly, bottom-right will have the largest sum x+y.
    #s = pts.sum(axis=1)
    s = np.sum(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # As clever as the previous observation, we can also
    # take the difference btw x and y
    # In [4]: A = np.array([[5, 10], [17, -9]])
    #    ...: A
    # Out[4]:
    # array([[ 5, 10],
    #        [17, -9]])
    # 
    # In [5]: np.diff(A, axis=1)
    # Out[5]:
    # array([[  5],
    #        [-26]])
    # So if we use np.diff(pts, axis=1), we'd get (y - x)'s
    # It's easy to prove that
    # the max of these is (bottom left)
    # the min of these is (top right)
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    tl, tr, br, bl = rect

    # Note that rect might not be rect-linear with the coordinate system.
    # Moreover, the widths (tr - tl) and (br - bl) might not be equal.
    # We shall take the max of these to make it the width of our destination rectangle.
    widthA = np.linalg.norm(tr - tl)
    widthB = np.linalg.norm(br - bl)
    widthMax = max(widthA, widthB)
    # Similary for the heights
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    heightMax = max(heightA, heightB)

    # The destination rectangle.
    # N.B. Must follow the prescribed order: tl, tr, br, bl
    dst = np.array([
        [0, 0],
        [widthMax-1, 0],
        [widthMax-1, heightMax-1],
        [0, heightMax-1],],
        dtype="float32")
    
    # Compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)

    ########################################################
    ## Code to help understand the perspective transform  ##
    ########################################################
    logging.debug(f"\nM.shape = {M.shape}\nM=\n{M}")
    rect_homogeneous = np.hstack((rect, np.ones((4,1))))
    #logging.debug(f"\nrect.shape = {rect.shape}\nrect=\n{rect}")
    #logging.debug(f"\nrect_homogeneous.shape = {rect_homogeneous.shape}\nrect_homogeneous=\n{rect_homogeneous}")
    #logging.debug(f"\nM@rect_homogeneous[0] = {M@rect_homogeneous[0]}\nM@rect_homogeneous[1] = {M@rect_homogeneous[1]}\nM@rect_homogeneous[2] = {M@rect_homogeneous[2]}\nM@rect_homogeneous[3] = {M@rect_homogeneous[3]}")

    logging.debug(f"""
M@rect_homogeneous[0] = {M@rect_homogeneous[0]}
M@rect_homogeneous[1] = {M@rect_homogeneous[1]}
M@rect_homogeneous[2] = {M@rect_homogeneous[2]}
M@rect_homogeneous[3] = {M@rect_homogeneous[3]}""")

    res_homogeneous = M @ rect_homogeneous.T
    res = res_homogeneous / res_homogeneous[2,:]
    logging.debug(f"\nres.shape = {res.shape}\nres=\n{res}")
    logging.debug(f"\nwidthMax = {widthMax}, heightMax = {heightMax}")
    # Now let's try the same thing with bulit-in functions
    #logging.debug(f"\ncv2.perspectiveTransform(rect, M) = {cv2.perspectiveTransform(rect, M)}")
    logging.debug(f"\ncv2.perspectiveTransform(rect[np.newaxis], M) =\n{cv2.perspectiveTransform(rect[np.newaxis], M)}")
    #logging.debug(f"\ncv2.perspectiveTransform(rect.T, M) = {cv2.perspectiveTransform(rect.T, M)}")
    #logging.debug(f"\ncv2.perspectiveTransform(rect_homogeneous, M) = {cv2.perspectiveTransform(rect_homogeneous, M)}")
    #logging.debug(f"\ncv2.perspectiveTransform(rect_homogeneous.T, M) = {cv2.perspectiveTransform(rect_homogeneous.T, M)}")
    ########################################################

    warped = cv2.warpPerspective(image, M, (widthMax, heightMax))
    # The last size arg of cv2.warpPerspective() is used to specify
    # the width and height of the output image, here warped

    return warped
