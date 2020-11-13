# Usage: sh shell/demo.sh
CUDA_VISIBLE_DEVICES=6 \
    python tools/demo.py \
    configs/pascal_voc/ssd300_voc.py \
    /data/sifan/model-zoo/pedestron/ssd300_voc_vgg16_caffe_240e_20190501-7160d09a.pth \
    /data/sifan/images/original/follow/webcam/ \
    /data/sifan/images/results/follow/webcam/ \
    --image_type jpg \
    --save_bbox False