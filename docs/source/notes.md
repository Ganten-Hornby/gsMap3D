# Notes

## Running Speed Optimization

### SSD for `memmap_tmp_dir`
Using `--memmap-tmp-dir` to point to an **SSD** would boost the reading speed of the `latent2gene` step, especially if you use a network device or HDD. This is due to the need for high-performance random reads when calculating the GSS (marker score matrix) using the gene rank matrix.

```bash
gsmap quick-mode \
    --memmap-tmp-dir "/path/to/local/ssd" \
    ...
```

## Use gpu/tpu

### `--use-gpu`
`gsMap` utilizes JAX for GPU/TPU acceleration. Ensure you have the appropriate JAX installation for your hardware.

### Jax preallocated memory
By default, JAX pre-allocates a large portion of GPU memory (around 75% or even 90% depending on the version/config). If you need to run multiple `gsMap` processes or other GPU tasks simultaneously, you can control this via environment variables:

```bash
# Disable pre-allocation
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Or set a specific memory fraction
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
```

### Use which device
If your machine has multiple GPUs, you can specify which device `gsMap` should use by setting the `CUDA_VISIBLE_DEVICES` environment variable:

```bash
# Use only the first GPU
export CUDA_VISIBLE_DEVICES=0

# Use the second GPU
export CUDA_VISIBLE_DEVICES=1
```
