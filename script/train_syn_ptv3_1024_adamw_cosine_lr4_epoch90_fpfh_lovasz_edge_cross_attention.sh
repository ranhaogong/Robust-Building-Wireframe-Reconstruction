CUDA_VISIBLE_DEVICES=1 \
python ../train_building3d.py \
--data_path /data/haoran/dataset/RoofReconstructionDataset/SyntheticDataset \
--scheduler cosine \
--optimizer adamw \
--max_ckpt_save_num 60 \
--cfg_file ../cfg/model_cfg_fpfh_lovasz_1024_cross_attention_p2rf.yaml \
--batch_size 64 \
--extra_tag syn_all_ptv3_1024_adamw_cosine_lr4_epoch90_fpfh_lovasz_edge_cross_attention_augment \
--epochs 90 \
--lr 1e-4

