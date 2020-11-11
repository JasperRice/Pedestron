# Usage: sh shell/demo.sh
CUDA_VISIBLE_DEVICES=2 \
    python tools/demo.py \
    configs/elephant/cityperson/cascade_hrnet.py \
    /data/sifan/model-zoo/pedestron/cascade_mask_rcnn_citypersons_hrnet_epoch_5.pth.stu \
    /data/sifan/images/original/follow/distance/webcam/ \
    /data/sifan/images/results/follow/distance/webcam/ \
    --image_type jpg \
    --save_bbox True