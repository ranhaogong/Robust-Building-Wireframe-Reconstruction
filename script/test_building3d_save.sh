CUDA_VISIBLE_DEVICES=1 \
python ../test_save_building3d.py \
--data_path "/data/haoran/dataset/building3d/Point2Roof" \
--cfg_file ../cfg/model_cfg_color_fpfh_2048_cross_attention.yaml \
--test_tag building3d_all_ptv3_color_2048_adamw_cosine_lr4_epoch150_fpfh_edge_cross_attention_augment \
--batch_size 64