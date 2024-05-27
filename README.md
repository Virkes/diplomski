# Master's Project
azure folder contains a Python program that generates captions and tags describing a picture, and finds objects inside a picture.
To run the program:
1. Install Microsoft Azure AI Vision Image Analysis for Python using `pip install azure-ai-vision-imageanalysis`
2. Run the program using `python caption.py` command in the files directory
 
darknet-yolo folder contains a Python program that finds objects inside a picture
To run the program:
1. Install OpenCV for Python using `pip install opencv-python`
2. Install darknet using
`git clone https://github.com/pjreddie/darknet`
`cd darknet`
`make`
3. Download the pre-trained weight file using `wget https://pjreddie.com/media/files/yolov3.weights`
4. Run the program using `python object_det.py`

db folder contains the results of the two programs run on NAPS database in JSON format