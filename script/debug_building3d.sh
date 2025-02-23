CUDA_VISIBLE_DEVICES=0 \
python -m debugpy --listen 5678 --wait-for-client ../train_building3d.py \
--data_path /data/haoran/dataset/building3d/Point2Roof \
--cfg_file ../cfg/model_cfg.yaml \
--batch_size 64 \
--extra_tag debug_13 \
--epochs 1 \
--lr 1e-3