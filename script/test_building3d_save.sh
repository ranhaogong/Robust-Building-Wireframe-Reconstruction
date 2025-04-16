CUDA_VISIBLE_DEVICES=1 \
python ../test_save_building3d.py \
--data_path /data/haoran/dataset/building3d/Point2Roof \
--cfg_file ../cfg/model_cfg_color_fpfh_lovasz_2048_cross_attention.yaml \
--test_tag building3dl_all_ptv3_color_2048_adamw_cosine_lr4_epoch150_fpfh_lovasz_edge_cross_attention_augment_gaussian_ablation \
--batch_size 64