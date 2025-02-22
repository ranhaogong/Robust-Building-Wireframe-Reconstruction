CUDA_VISIBLE_DEVICES=1 \
python ../train_building3d.py \
--data_path /data/haoran/dataset/building3d/Point2Roof \
--max_ckpt_save_num 60 \
--cfg_file ../cfg/model_cfg_color_2048.yaml \
--batch_size 64 \
--extra_tag building3d_all_ptv3_color_2048_epoch150 \
--epochs 150 \
--lr 1e-3

