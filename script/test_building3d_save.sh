CUDA_VISIBLE_DEVICES=0 \
python ../test_save_building3d.py \
--data_path /data/haoran/dataset/building3d/roof/Tallinn/train/ \
--cfg_file ../cfg/model_cfg_2048.yaml \
--test_tag building3d_trainset_pointnet2_2048 \
--batch_size 64