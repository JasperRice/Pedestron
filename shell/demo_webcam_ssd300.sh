# Usage: sh shell/webcam.sh
CUDA_VISIBLE_DEVICES=0 \
    python tools/demo.py \
    configs/pascal_voc/ssd300_voc.py \
    /home/midea/Documents/model_zoo/pedestron/ssd300_voc_vgg16_caffe_240e_20190501-7160d09a.pth \
    --image_save_path /home/midea/Pictures/demo