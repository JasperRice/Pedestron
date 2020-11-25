# Usage: sh shell/demo.sh
CUDA_VISIBLE_DEVICES=0 \
    python tools/demo.py \
    configs/elephant/cityperson/cascade_hrnet.py \
    /cv_data/sifan/model-zoo/pedestron/cascade_mask_rcnn_caltech_hrnet_epoch_14.pth.stu \
    --input_img_dir /home/midea/Documents/Pedestron/demo \
    --output_dir /home/midea/Documents/Pedestron/result_demo \
    --image_type png