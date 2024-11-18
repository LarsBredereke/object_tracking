# Object Tracking Pipeline
This repo contains a modular Pipeline for tracking objects in 3D using rgb cameras.
It is made of python scripts that each read from and write to the 'data' directory.
The whole pipeline can be executed using the main.sh script.

## Step-by-step instructions

### YOLO model
Train a yolo model to detect your object classes. For this example, you can just use the provided one.

### Video Footage
Upload video files into data/<session>/<trial> directory. Make sure the n-th frame of video a is from the same point in time as the n-th frame of video b for all pairs of videos a,b. For this Example, just leave the example videos as is.

### Initial Guess of Camera Parameters
If your cameras are positioned differently from those of the example videos, you need to update the initial guess of their poses in config.py

### Start the Pipeline
add the session and trial ids to be processed to config.py. For this example this is session 28, trial 1
Start main.sh. The annotation tool will open and display a camera's image. If you have a different setup than in the example, you need to modify annotate_corners.py. Otherwise, follow these steps:
- Press 't' to annotate the table corners. Refer to annotate_example.png and press 'q', then click on the upper left corner of the table. press 'e' and click on the upper right corner, 'd' and lower right, and 'a' and lower left corner.
- Press 'c' and annotate the counter (the narrow table) likewise.
- Press 'o' and click on the world origin marker on the floor between the tables.
- Note: invisible features can be skipped. However, at least 4 features need to be annotated for each image.
- Redo annotation if necessary using 't', 'c' or 'o' until you have annotated all visible features accurately.
- Press 's' to move on to the next image, 'r' to redo the previous image, or 'x' to skip this image if it is not annotatable. Note: which corner is 'upper left' does not depend on the camera perspective.

After annotation, the rest of the pipeline does not need manual input. The trajectories are written into the the data/<session>/<trial> directory.