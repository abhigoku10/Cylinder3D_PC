name=cylinder_asym_networks
gpuid=0

CUDA_VISIBLE_DEVICES=${gpuid}  python3 -u test_cylinder_custom_sk.py 
2>&1 | tee logs_dir/${name}_logs_tee.txt
