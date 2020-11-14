# Usage: sh shell/demo_webcam_cascade_mobilenet.sh
CUDA_VISIBLE_DEVICES=0 \
    python tools/demo.py \
    /home/midea/Documents/Pedestron/configs/elephant/cityperson/cascade_mobilenet.py \
    /home/midea/Documents/model_zoo/pedestron/cascade_mask_rcnn_citypersons_mobilenet_epoch_16.pth.stu