CUDA_VISIBLE_DEVICES=0 \
python ../test_save_building3d.py \
--data_path /data/haoran/dataset/RoofReconstructionDataset/RealDataset \
--cfg_file ../cfg/model_cfg_2048_p2rf.yaml \
--test_tag building3d_all_pointnet2_2048_real \
--batch_size 64