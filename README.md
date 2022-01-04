# Image-Segementation
Run the following Code for Inference

#################################################

LINK FOR .h5 FILE

https://drive.google.com/file/d/1EVy2I6S54T4GwkFkx6aY54R5-Fm7TZyY/view?usp=sharing 
######################################################

!pip3 install pixellib

import pixellib

from pixellib.instance import custom_segmentation

test_video = custom_segmentation()
test_video.inferConfig(num_classes=  2, class_names=["BG", "light", "laptop"])
test_video.load_model("/content/drive/MyDrive/ALL_TASK/instant_seg/mask_rcnn_model.007-0.740388.h5")
test_video.process_video("/content/my_video.mp4", show_bboxes = True,  output_video_name="video_o1.mp4", frames_per_second=2)


INPUT:
![alt text](https://github.com/Snehadhole/Image-Segementation/blob/main/10piclap6.jpeg.png?raw=true)



