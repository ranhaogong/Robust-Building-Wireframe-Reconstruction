CUDA_VISIBLE_DEVICES=3 \
python ../test_save_building3d.py \
--data_path /data/haoran/dataset/building3d/Point2Roof_tokyo_seg \
--cfg_file ../cfg/model_cfg_color_mrgd_lovasz_2048_dbscan_003.yaml \
--test_tag building3d_tokyo_all_ptv3_color_2048_adamw_cosine_lr4_epoch150_mrgd_lovasz_edge_dbscan_003 \
--batch_size 64