python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/ssd_inception_v2_original_300x300.config --trained_checkpoint_prefix training/model.ckpt-200000 --output_directory trained-inference-graphs/output_inference_graph_300x300_sc.pb




# for small images

python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/ssd_inception_v2_coco.config --trained_checkpoint_prefix /home/nijeri/Documents/training_small_pictures/model.ckpt-200000 --output_directory trained-inference-graphs/output_inference_graph_small_pictures.pb
