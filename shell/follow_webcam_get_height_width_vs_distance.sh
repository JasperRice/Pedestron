# Usage: sh shell/follow_webcam_get_height_width_vs_distance.sh
CUDA_VISIBLE_DEVICES=6 \
    python tools/demo.py \
    configs/elephant/cityperson/cascade_hrnet.py \
    /cv_data/sifan/model-zoo/pedestron/cascade_mask_rcnn_citypersons_hrnet_epoch_5.pth.stu \
    --input_img_dir /cv_data/sifan/images/original/follow/distance/dw800/ \
    --output_dir /cv_data/sifan/images/results/follow/distance/dw800/ \
    --image_type jpg \
    --save_bbox True