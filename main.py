import datetime
import os
import time
import warnings

import util.presets as presets
import torch
import torch.utils.data
import torchvision
import util.transforms as transforms
import util.utils_backbone as utils
from util.sampler import RASampler 
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode
import torchvision.datasets as datasets
from torchvision.models import resnet18, ResNet18_Weights, inception_v3, Inception_V3_Weights, vit_b_16,ViT_B_16_Weights, ConvNeXt_Tiny_Weights
from model.base_model import convnext_tiny
import model.op_model as op_model
import util.cub_200_2011 as cub200

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

class Combined_model(nn.Module):
    def __init__(self, modelA, modelB):
        super(Combined_model, self).__init__()
        self.model_op = modelA
        self.model_backbone = modelB
        
    def forward(self, x):
        if args.model == 'inception_v3':
            out = self.model_op(x, 299)
        else:
            out = self.model_op(x, 224)
        out = self.model_backbone(out)
        return out

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema=None, scaler=None):
    if args.second_phase:
        model.module.model_op.train()
        model.module.model_backbone.eval()
        if args.model == 'inception_v3':
            model.module.model_backbone.training = True
    else:
        model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    for i,(pair) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        start_time = time.time()
        losses = []
        outputs = []
        targets = []
        for j in range(len(pair)):
            image, target = pair[j]
            image, target = image.to(device), target.to(device)
            if args.operator == 'None':
                image = torch.nn.functional.interpolate(image, size=(args.train_crop_size,args.train_crop_size), mode='bicubic', align_corners=True)
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                if args.model == 'inception_v3':
                    output, aux_output = model(image)
                    loss1 = criterion(output, target)
                    loss2 = criterion(aux_output, target)
                    loss = loss1 + 0.4*loss2
                else:
                    output = model(image)
                    loss = criterion(output, target)
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                if args.clip_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()

            if model_ema and i % args.model_ema_steps == 0:
                model_ema.update_parameters(model)
                if epoch < args.lr_warmup_epochs:
                    model_ema.n_averaged.fill_(0)
            losses.append(loss.reshape(1))
            outputs.append(output)
            targets.append(target)
        loss = torch.cat([*losses], dim=0).mean()
        output = torch.cat([*outputs], dim = 0)
        target = torch.cat([*targets], dim = 0)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))


def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix=""):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            if args.operator == 'None':
                image = torch.nn.functional.interpolate(image, size=(args.train_crop_size,args.train_crop_size), mode='bicubic', align_corners=True)
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size
    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
    return metric_logger.acc1.global_avg

def load_data(datadir, args, multi):
    # Data loading code
    print("Loading data")
    val_resize_size, val_crop_size, train_crop_size = args.val_resize_size, args.val_crop_size, args.train_crop_size
    interpolation = InterpolationMode(args.interpolation)
    
    print("Loading training data")
    st = time.time()
    auto_augment_policy = getattr(args, "auto_augment", None)
    random_erase_prob = getattr(args, "random_erase", 0.0)
    transform = presets.ClassificationPresetTrain(
            crop_size=train_crop_size,
            interpolation=interpolation,
            auto_augment_policy=auto_augment_policy,
            random_erase_prob=random_erase_prob,
        )
    if args.dataset == 'imagenet':
        traindir = os.path.join(datadir,'imagenet224/train')
        dataset = torchvision.datasets.ImageFolder(traindir, transform)
    elif args.dataset == 'flowers':
        dataset = datasets.Flowers102(root = datadir, transform=transform, split='train', download=False)
    elif args.dataset == 'cars':
        dataset = datasets.StanfordCars(root = datadir, transform=transform, split='train', download=True)
    elif args.dataset == 'pets':
        dataset = datasets.OxfordIIITPet(root = datadir, transform=transform, split='trainval', download=False)
    elif args.dataset == 'aircraft':
        dataset = datasets.FGVCAircraft(root = datadir, transform=transform, split='train', download=False)
    elif args.dataset == 'food':
        dataset = datasets.Food101(root = datadir, transform=transform, split='train', download=False)
    elif args.dataset == 'cub200':
        dataset = cub200.Cub2011(root = datadir, train=True, transform=transform, download=False)
    else:
        print('Not implemented dataset : {}'.format(args.dataset))
    print("Took", time.time() - st)

    if multi:
        print("Loading validation data")
        transform = presets.ClassificationPresetEval(
            crop_size=val_crop_size, resize_size=val_resize_size, interpolation=interpolation
        )
        if args.dataset == 'imagenet':
            valdir = os.path.join(datadir,'imagenet224/val')
            dataset_test = torchvision.datasets.ImageFolder(valdir, transform)
        elif args.dataset == 'flowers':
            dataset_test = datasets.Flowers102(root = datadir, transform=transform, split='test', download=False)
        elif args.dataset == 'cars':
            dataset_test = datasets.StanfordCars(root = datadir, transform=transform, split='test', download=True)
        elif args.dataset == 'pets':
            dataset_test = datasets.OxfordIIITPet(root = datadir, transform=transform, split='test', download=False)
        elif args.dataset == 'aircraft':
            dataset_test = datasets.FGVCAircraft(root = datadir, transform=transform, split='test', download=False)
        elif args.dataset == 'food':
            dataset_test = datasets.Food101(root = datadir, transform=transform, split='test', download=False)
        elif args.dataset == 'cub200':
            dataset_test = cub200.Cub2011(root = datadir, train=False, transform=transform, download=False)
        else:
            print('Not implemented dataset : {}'.format(args.dataset))
        
        print("Creating data loaders")
        if args.distributed:
            if hasattr(args, "ra_sampler") and args.ra_sampler:
                train_sampler = RASampler(dataset, shuffle=True, repetitions=args.ra_reps)
            else:
                train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
        else:
            train_sampler = torch.utils.data.RandomSampler(dataset)
            test_sampler = torch.utils.data.SequentialSampler(dataset_test)
        return dataset, dataset_test, train_sampler, test_sampler
    else:
        if args.distributed:
            if hasattr(args, "ra_sampler") and args.ra_sampler:
                train_sampler = RASampler(dataset, shuffle=True, repetitions=args.ra_reps)
            else:
                train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            train_sampler = torch.utils.data.RandomSampler(dataset)
        return dataset

def get_custom_collate(mixupcutmix):
    def custom_collate(batch):
        batches = []
        for i in range(len(batch[0])):
            batch_list = []
            for j in range(len(batch)):
                batch_list.append(batch[j][i])
            batches.append(default_collate(batch_list))
        return mixupcutmix(tuple(batches))
    return custom_collate

def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
    if args.test_only:
        if args.model == 'inception_v3':
            train_size=[299]
        else:
            train_size=[224]
        args.train_crop_size = train_size[0]
        _, dataset_test, train_sampler, test_sampler = load_data(args.data_path, args, True)
    else:
        assert args.train_data_size, "Argument Missing : Train data sizes"
        train_size = [int(i) for i in args.train_data_size.split(',')]
        if args.model == 'inception_v3':
            assert train_size[0]==299, "The first resolution of the train size should be 299"
        else:
            assert train_size[0]==224, "The first resolution of the train size should be 224"
        dataset_list = []
        for i in range(len(train_size)):
            args.train_crop_size = train_size[i]
            if i==0:
                dataset, dataset_test, train_sampler, test_sampler = load_data(args.data_path, args, True)
            else:
                dataset = load_data(args.data_path, args, False)
            dataset_list.append(dataset)
        if args.model == 'inception_v3':
            args.train_crop_size = 299
        else:
            args.train_crop_size = 224
            
    collate_fn = None
    if (args.dataset == 'imagenet') or (args.dataset == 'cars') or (args.dataset == 'pets') or (args.dataset == 'aircraft') or (args.dataset == 'food'):
        num_classes = len(dataset_test.classes)
    elif args.dataset == 'flowers':
        num_classes = 102
    elif args.dataset == 'cub200':
        num_classes = 200
    else:
        pass
    
    mixup_transforms = []
    if args.mixup_alpha > 0.0:
        mixup_transforms.append(transforms.RandomMixup(num_classes, p=1.0, alpha=args.mixup_alpha))
    if args.cutmix_alpha > 0.0:
        mixup_transforms.append(transforms.RandomCutmix(num_classes, p=1.0, alpha=args.cutmix_alpha))
    if mixup_transforms:
        mixupcutmix = torchvision.transforms.RandomChoice(mixup_transforms)
        collate_fn = get_custom_collate(mixupcutmix)
    
    if not args.test_only:
        data_loader = torch.utils.data.DataLoader(
            ConcatDataset(*dataset_list),
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True
    )

    print("Creating model")
    if args.model == "inception_v3":
        print("Getting inception_v3 pre-trained model")
        weights = Inception_V3_Weights.IMAGENET1K_V1
        model_back = inception_v3(weights=weights)
        print('Backbone total params: %.2fM' % (sum(p.numel() for p in model_back.parameters())/1000000.0))
    elif args.model == "resnet18":
        print("Getting resnet18 pre-trained model")
        weights = ResNet18_Weights.IMAGENET1K_V1
        model_back = resnet18(weights=weights)
        print('Backbone total params: %.2fM' % (sum(p.numel() for p in model_back.parameters())/1000000.0))
    elif args.model == "convnext":
        print("Getting convnext pre-trained model")
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        model_back = convnext_tiny(weights=weights)
        if args.dataset != 'imagenet':
            model_back.classifier[2] = nn.Linear(model_back.classifier[2].in_features, num_classes)
        print('Backbone total params: %.2fM' % (sum(p.numel() for p in model_back.parameters())/1000000.0))
    elif args.model == "vit_b_16":
        print("Getting vit_b_16 pre-trained model")
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        model_back = vit_b_16(weights=weights)
        print('Backbone total params: %.2fM' % (sum(p.numel() for p in model_back.parameters())/1000000.0))
    else:
        print('Not implemented model : {}'.format(args.model))

    if args.operator == 'None':
        model = model_back
    elif args.operator == 'pfno':
        print("Getting operator")
        if args.model == "inception_v3":
            model_op = op_model.lfno_layer(150)
        else:
            model_op = op_model.lfno_layer(113)
        print('Operator total params: %.2fM' % (sum(p.numel() for p in model_op.parameters())/1000000.0))
        model = Combined_model(model_op,model_back)
    elif args.operator == 'fno':
        print("Getting operator")
        if args.model == "inception_v3":
            model_op = op_model.FNO2d(150)
        else:
            model_op = op_model.FNO2d(113)
        print('Operator total params: %.2fM' % (sum(p.numel() for p in model_op.parameters())/1000000.0))
        model = Combined_model(model_op,model_back)
    elif args.operator == 'pfno_large':
        print("Getting operator")
        if args.model == "inception_v3":
            model_op = op_model.lfno_layer_large(150)
        else:
            model_op = op_model.lfno_layer_large(113)
        print('Operator total params: %.2fM' % (sum(p.numel() for p in model_op.parameters())/1000000.0))
        model = Combined_model(model_op,model_back)
    elif args.operator == 'pfno_small':
        print("Getting operator")
        if args.model == "inception_v3":
            model_op = op_model.lfno_layer_small(150)
        else:
            model_op = op_model.lfno_layer_small(113)
        print('Operator total params: %.2fM' % (sum(p.numel() for p in model_op.parameters())/1000000.0))
        model = Combined_model(model_op,model_back)
    else:
        print('Not implemented operator : {}'.format(args.operator))
        exit()
    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))
    parameters = utils.set_weight_decay(
        model,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    model_ema = None
    if args.model_ema:
        # Decay adjustment that aims to keep the decay independent from other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and ommit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        if args.fine_grain_tuned:
            model_without_ddp.model_backbone.load_state_dict(checkpoint["model"])
        else:
            model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.test_only:
            if (not args.second_phase) and (not args.fine_grain_tuned) and (not args.reset_lr):
                optimizer.load_state_dict(checkpoint["optimizer"])
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        if (not args.second_phase) and (not args.fine_grain_tuned):
            args.start_epoch = checkpoint["epoch"] + 1
        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if model_ema:
            evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
        else:
            evaluate(model, criterion, data_loader_test, device=device)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema, scaler)
        lr_scheduler.step()
        evaluate(model, criterion, data_loader_test, device=device)
        if model_ema:
            evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            if model_ema:
                checkpoint["model_ema"] = model_ema.state_dict()
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    parser.add_argument("--data-path", default='./data', type=str, help="dataset path")
    parser.add_argument("--dataset", default="imagenet", type=str, help="dataset name [imagenet, flowers, cars, pets, aircraft, food, cub200]")
    parser.add_argument("--model", default="resnet18", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--epochs", default=90, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.001, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--bias-weight-decay",
        default=None,
        type=float,
        help="weight decay for bias parameters of all layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--transformer-embedding-decay",
        default=None,
        type=float,
        help="weight decay for embedding parameters for vision transformer models (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing"
    )
    parser.add_argument("--mixup-alpha", default=0.0, type=float, help="mixup alpha (default: 0.0)")
    parser.add_argument("--cutmix-alpha", default=0.0, type=float, help="cutmix alpha (default: 0.0)")
    parser.add_argument("--lr-scheduler", default="steplr", type=str, help="the lr scheduler (default: steplr)")
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument(
        "--lr-warmup-method", default="constant", type=str, help="the warmup method (default: constant)"
    )
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr-step-size", type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output-dir", default="./checkpoints", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
    parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument(
        "--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters"
    )
    parser.add_argument(
        "--model-ema-steps",
        type=int,
        default=32,
        help="the number of iterations that controls how often to update the EMA model (default: 32)",
    )
    parser.add_argument(
        "--model-ema-decay",
        type=float,
        default=0.99998,
        help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",
    )
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )
    parser.add_argument(
        "--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)"
    )
    parser.add_argument(
        "--val-resize-size", type=int, help="the resize size used for validation"
    )
    parser.add_argument(
        "--val-crop-size", type=int, help="the central crop size used for validation"
    )
    parser.add_argument(
        "--train-crop-size", type=int, help="the random crop size used for training"
    )
    parser.add_argument("--clip-grad-norm", default=None, type=float, help="the maximum gradient norm (default None)")
    parser.add_argument("--ra-sampler", action="store_true", help="whether to use Repeated Augmentation in training")
    parser.add_argument(
        "--ra-reps", default=3, type=int, help="number of repetitions for Repeated Augmentation (default: 3)"
    )
    parser.add_argument("--train-data-size", type=str, help="the data sizes to train. original size must come first")
    parser.add_argument("--second-phase", action="store_true", help="whether to reset learning rate and epoch")
    parser.add_argument("--operator", default="None", type=str, help="model name")
    parser.add_argument("--fine-grain-tuned", action="store_true", help="loading checkpoint for fine-tuned backbone model with fine-grained datset")
    parser.add_argument("--reset-lr", action="store_true", help="resetting learning rate and epoch")
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)