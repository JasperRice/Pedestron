# Usage: sh shell/demo.sh
CUDA_VISIBLE_DEVICES=7 \
    python tools/demo.py \
    configs/fp16/faster_rcnn_r50_fpn_fp16_1x.py \
    /data/sifan/model-zoo/pedestron/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth \
    /data/sifan/images/original/follow/webcam/ \
    /data/sifan/images/results/follow/webcam/ \
    --image_type jpg \
    --save_bbox False