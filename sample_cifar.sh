PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u main.py \
    --config cifar10.yml \
    --exp experiments/cifar10_sampling \
    --doc /path/to/your/downloaded/model-2388000.ckpt \
    --sample --fid --timesteps 100 --eta 0 --ni \
    --image_folder results/cifar10_samples \
    --skip_type quad \
    --bitwidth 6 \
    --calib_t_mode diff \
    --batch_size 1 \

# example - running cifaar10 dataset sampling