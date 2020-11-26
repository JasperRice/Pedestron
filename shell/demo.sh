# Usage: sh shell/demo.sh
CUDA_VISIBLE_DEVICES=0 \
    python tools/demo.py \
    configs/elephant/cityperson/cascade_hrnet.py \
    models/cascade_mask_rcnn_citypersons_hrnet_epoch_5.pth.stu \
    --input_img_dir /home/midea/Documents/Pedestron/demo \
    --output_dir /home/midea/Documents/Pedestron/result_demo \
    --image_type png