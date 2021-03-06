## Origin
Around January 2020, I was implementing this social distancing program that many of us first saw on [Landing.ai's announcement](https://landing.ai/landing-ai-creates-an-ai-tool-to-help-customers-monitor-social-distancing-in-the-workplace/).

At that time, on the Internet there were not many people sharing how exactly they did it. I found two or three github repos, but none of them was satisfying (_Well, strictly speaking, there was one interesting repo which used **OpenPose** and **body part length** to conceive a **two-meter distance**_.).

It was about the same time when [pyimagesearch](https://www.pyimagesearch.com/2020/06/01/opencv-social-distancing-detector/) published an article on the same topic. It was a good article, but it only slightly hinted at how it would further improve in the **bird's-eye view** direction 


## Example results
The following video is downloaded from and credited to [some famous website on MOT](https://motchallenge.net/vis/MOT17-02-SDP).
<br/>

![example](figs/example.gif "mot-video")

<br/>

And the next one shows an example of how to apply one of the scripts to an image: _Users of this repo should **mark four points** like shown in this screenshot, either **clockwisely** or **counter-clockwisely**_

<br/>

![example2](figs/example2-one-punch-man.png "one-punch")


## How to use this repo
01. Create a folder named `yolo-coco`
02. Download the configuration and weight files to `yolo-coco` folder
    ```bash
    # from Joseph Redmon's github repo
    curl -o yolo-coco/yolov3.cfg https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
    curl -o yolo-coco/yolov3.weights https://pjreddie.com/media/files/yolov3.weights
    curl -o yolo-coco/coco.names https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
    ```
Once having all the files for YOLO to run, just run any one of the python scripts according to the usage described
in each file.
- `social_distance_detector.py`, `single_frame_detector.py` are scripts coming from pyimagesearch
- `webcam_record.py` just provides a way to record a video from the webcam
- `street_view_04_.py` is an intermediate test script
- `image_BEV.py`, `video_BEV.py` both produce **Bird's Eye View**, one for image, the other for video
	- These scripts when run will first open up a image window called `Draw` for the user to mark four points on, which correspond to a rectangle in the real world represented by the image. This rectangle is used to construct a perspective change of the ground on which people in the video/image walk.

For those of you who really want to get your hands dirty and to play with this repo: You might find useful to tweak the parameters inside the script **`pyimagesearch/social_distancing_config.py`**.


## Environment
This repo uses Python3. I think Python2 might just work as well, but the provided environment files might not allow you to install everything correctly out of the box.

It's recommended to use a virtual environment (virtualenv, miniconda, etc.). A `requirements.txt` file is provided
below for `virtualenv` (or for `virtualenvwrapper` for that matter), and a `environment.yml` for `miniconda`.
- so either you don't use `miniconda`, then simply
    ```bash
    pip install -r requirements.txt
    ```
- or you're `miniconda` user, then
    ```bash
    conda env create -f environment.yml
    # Then you should activate the environment before being able to run the scripts in this repo
    conda activate sociald
    ```
    - I have also created another `environment_heavier.yml` because actually I was not familiar with miniconda, not knowing which way of installing the packages was the best practice. I will leave `environment_heavier.yml` there just in case.


## Difficulties
At the time of implementing this, I only slightly heard of **perspective change** but did not know exactly how to do it.

The main idea of this repo is simple:
01. Use **YOLO** to obtain bounding boxes on persons
02. Take one representative point from each bounding box (my choice being the **center bottom** point)
03. Use **perspective change** to rectify the plane on which the persons walk in the video, and calculate the distance between the transformed representative points

Maybe I will write a small repo on perspective change later. For the moment being, I would just point the interested readers to
- [a fine medium article](https://medium.com/acmvit/how-to-project-an-image-in-perspective-view-of-a-background-image-opencv-python-d101bdf966bc)
- [pyimagesearch's article on perspective change](https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/)
- [a decent book titled "Multiple View Geometry in Computer Vision"](https://www.robots.ox.ac.uk/~vgg/hzbook/)

I finished the code collected in this repo _around February 2020_ and have not polished it ever since. Sorry for the delay and hope what I shared here might be useful.


