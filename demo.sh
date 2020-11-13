CUDA_VISIBLE_DEVICES=0 python tools/demo.py \
    configs/elephant/cityperson/faster_rcnn_hrnet.py \
    models/faster_rcnn_cityperosns_hrnet_epoch_1.pth.stu \
    demo/ \
    result_demo/ \
    --image_type 'png'