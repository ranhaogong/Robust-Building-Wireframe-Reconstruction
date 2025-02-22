CUDA_VISIBLE_DEVICES=1 \
python ../test_save_building3d.py \
--data_path /data/haoran/dataset/building3d/Point2Roof \
--cfg_file ../cfg/model_cfg_color_2048.yaml \
--test_tag building3d_all_ptv3_color_2048_epoch150 \
--batch_size 4