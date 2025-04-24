CUDA_VISIBLE_DEVICES=3 \
python ../train_building3d.py \
--data_path /data/haoran/dataset/building3d/Point2Roof_tokyo_seg \
--scheduler cosine \
--optimizer adamw \
--max_ckpt_save_num 60 \
--cfg_file ../cfg/model_cfg_color_fpfh_mrgd_lovasz_2048_tokyo_dbscan_004.yaml \
--batch_size 64 \
--extra_tag building3d_tokyo_all_ptv3_color_2048_adamw_cosine_lr4_epoch150_fpfh_cross_attention_lovasz_edge_dbscan_004 \
--epochs 150 \
--lr 1e-4

