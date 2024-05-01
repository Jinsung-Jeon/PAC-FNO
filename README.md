# PAC-FNO
## Requirements
conda env create --file environment.yaml

## Dataset
We provide our training and testing codes of our method for Inception-V3 trained with ImageNet-1K and ConvNeXt-Tiny trained with Stanford Cars.
The parser (``--data-path``) means where datasets exist and the default is ``./dataset``.
Stanford Cars will be automatically downloaded if you execute ``main.py``, while ImageNet should be downloaded manually from https://www.image-net.org/ .

## Train

### Inception-V3 with ImageNet-1K
#### First traning step
```bash
torchrun --nproc_per_node=4 main.py --model inception_v3 --dataset imagenet --operator pfno --val-resize-size 341 --val-crop-size 299 --train-crop-size 299 --lr 0.001 --lr-step-size 20 --output-dir ./checkpoints/inception_imagenet_first_phase/ --train-data-size 299
```
#### Second traning step
```bash
torchrun --nproc_per_node=4 main.py --model inception_v3 --dataset imagenet --operator pfno --val-resize-size 341 --val-crop-size 299 --train-crop-size 299 --lr 1e-6 --lr-step-size 10 --output-dir ./checkpoints/inception_imagenet_second_phase/ --resume ./checkpoints/inception_imagenet_first_phase/model_{}.pth --train-data-size 299,32,64,128 --second-phase
```

### ConvNeXt-Tiny with Stanford Cars
#### Fine-tuneing the model pre-trained on Imagenet-1K with Stanford Cars
```bash
torchrun --nproc_per_node=4 main.py --model convnext --dataset cars --batch-size 128 --opt adamw --lr 1e-3 --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --auto-augment ta_wide --epochs 600 --random-erase 0.1 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 --weight-decay 0.05 --norm-weight-decay 0.0 --val-resize-size 256 --val-crop-size 224 --train-crop-size 224 --ra-sampler --ra-reps 4 --output-dir ./checkpoints/convnext_cars_fine --train-data-size 224
```
#### First traning step
```bash
torchrun --nproc_per_node=4 main.py --model convnext --operator pfno_large --dataset cars --batch-size 128 --opt adamw --lr 1e-3 --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --auto-augment ta_wide --epochs 600 --random-erase 0.1 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 --weight-decay 0.05 --norm-weight-decay 0.0 --val-resize-size 256 --val-crop-size 224 --train-crop-size 224 --ra-sampler --ra-reps 4 --resume ./checkpoints/convnext_cars_fine/model_{}.pth --output-dir ./checkpoints/convnext_cars_first_phase --train-data-size 224 --fine-grain-tuned
```
#### Second traning step
```bash
torchrun --nproc_per_node=4 main.py --model convnext --operator pfno_large --dataset cars --batch-size 128 --opt adamw --lr 1e-6 --lr-scheduler cosineannealinglr --lr-warmup-epochs 5 --lr-warmup-method linear --auto-augment ta_wide --epochs 600 --random-erase 0.1 --label-smoothing 0.1 --mixup-alpha 0.2 --cutmix-alpha 1.0 --weight-decay 0.05 --norm-weight-decay 0.0 --val-resize-size 256 --val-crop-size 224 --train-crop-size 224 --ra-sampler --ra-reps 4 --resume ./checkpoints/convnext_cars_first_phase/model_{}.pth --output-dir ./checkpoints/convnext_cars_second_phase --train-data-size 224,32,64,128 --second-phase
```

``--nproc_per_node`` means the number of gpus to use. 

``--resume`` the best checkpoints from the former phase.

## Test
We released checkpoints for inception v3 on imagenet.
To reproduce our results, ``--val-resize-size`` and ``--val-crop-size`` should be adjusted.
The following sets of sizes are used for evaluation.

| val resize size | 36 | 73 | 146 | 256 | 341 |
|-----------------|----|----|-----|-----|-----|
| val crop size   | 32 | 64 | 128 | 224 | 299 |

```bash
torchrun --nproc_per_node=4 main.py --model inception_v3 --dataset imagenet --operator pfno_small --val-resize-size 341 --val-crop-size 299 --test-only --resume ./checkpoints/inception_imagenet/model.pth
```
```bash
torchrun --nproc_per_node=4 main.py --model convnext --dataset cars --operator pfno_large --val-resize-size 256 --val-crop-size 224 --test-only --resume ./checkpoints/convnext_cars_second_phase/model_{}.pth
```
