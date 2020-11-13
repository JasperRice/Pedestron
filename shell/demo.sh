# Usage: sh shell/demo.sh
CUDA_VISIBLE_DEVICES=6 \
    python tools/demo.py \
    configs/elephant/cityperson/cascade_hrnet.py \
    /data/sifan/model-zoo/pedestron/cascade_mask_rcnn_citypersons_hrnet_epoch_5.pth.stu \
    demo/ \
    result_demo/ \
    --image_type 'png'