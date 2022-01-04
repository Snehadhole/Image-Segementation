# Image-Segementation
Run the following Code for Inference
!pip3 install pixellib
import pixellib
from pixellib.instance import custom_segmentation

test_video = custom_segmentation()
test_video.inferConfig(num_classes=  2, class_names=["BG", "light", "laptop"])
test_video.load_model("/content/drive/MyDrive/ALL_TASK/instant_seg/mask_rcnn_model.007-0.740388.h5")
test_video.process_video("/content/my_video.mp4", show_bboxes = True,  output_video_name="video_o1.mp4", frames_per_second=2)
