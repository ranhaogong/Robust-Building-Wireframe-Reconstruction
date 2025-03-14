CUDA_VISIBLE_DEVICES=0 \
python ../test_save_building3d.py \
--data_path /data/haoran/Point2Roof/testmydata \
--cfg_file ../cfg/model_cfg_color_lovasz_2048.yaml \
--test_tag testmydata \
--batch_size 64