from datasets import Iterator
from .utils import Average_Meter, Timer, print_and_save_log, mIoUOnline, get_numpy_from_tensor, save_model, write_log, \
    check_folder, one_hot_embedding_3d
import torch
import cv2
import torch.nn.functional as F
import os
import torch.nn as nn
from torch.amp import autocast, GradScaler
import wandb


class BaseRunner():
    def __init__(self, model, optimizer, losses, train_loader, val_loader, scheduler):
        self.optimizer = optimizer
        self.losses = losses
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.scheduler = scheduler
        self.train_timer = Timer()
        self.eval_timer = Timer()
        try:
            use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
        except KeyError:
            use_gpu = '0'
        self.the_number_of_gpu = len(use_gpu.split(','))
        self.original_size = self.model.img_adapter.sam_img_encoder.img_size
        if self.the_number_of_gpu > 1:
            self.model = nn.DataParallel(self.model)


class SemRunner(BaseRunner):
    # def __init__(self, **kwargs):
    #     super().__init__(kwargs)

    def __init__(self, model, optimizer, losses, train_loader, val_loader, scheduler):
        super().__init__(model, optimizer, losses, train_loader, val_loader, scheduler)
        self.exist_status = ['train', 'eval', 'test']
        self.scaler = GradScaler("cuda")  # Initialize with device type

    def init_wandb(self, cfg):
        """Initialize wandb with project configuration"""
        wandb.init(
            project="SAM_sem_seg",  # Project name
            config={
                "architecture": "Modified-SAM",
                "dataset": cfg.dataset.name if hasattr(cfg.dataset, 'name') else "custom",
                "learning_rate": cfg.opt_params.lr_default,
                "learning_rate_list": cfg.opt_params.lr_list,
                "weight_decay": cfg.opt_params.wd_default,
                "weight_decay_list": cfg.opt_params.wd_list,
                "batch_size": cfg.bs,
                "optimizer": cfg.opt_name,
                "scheduler": cfg.scheduler_name,
                "scheduler_params": {
                    "warmup_factor": cfg.scheduler_params.warmup_factor,
                    "warmup_steps": cfg.scheduler_params.warmup_steps,
                    "milestones": cfg.scheduler_params.stepsize,
                    "gamma": cfg.scheduler_params.gamma
                },
                "max_iterations": cfg.max_iter,
                "num_classes": cfg.model.params.class_num,
                "model_type": cfg.model.params.model_type,
                "image_size": cfg.dataset.transforms.resize.params.size,
                "loss_config": {
                    name: {
                        "weight": params.weight,
                        **params.params
                    } for name, params in cfg.losses.items()
                }
            }
        )
        # Watch model to track gradients and parameters
        wandb.watch(self.model, log="all", log_freq=100)

    def log_metrics(self, metrics, step, phase="train"):
        """Log metrics to wandb with proper prefixing"""
        wandb_metrics = {f"{phase}/{k}": v for k, v in metrics.items()}
        wandb_metrics["step"] = step
        wandb.log(wandb_metrics)

    def log_images(self, images, masks_pred, labels, step, max_images=4):
        """Log image predictions to wandb"""
        if images.shape[0] == 0:
            return
        
        # Take only the first few images
        n_images = min(images.shape[0], max_images)
        images = images[:n_images]
        masks_pred = masks_pred[:n_images]
        labels = labels[:n_images]

        # Convert predictions to class indices
        pred_labels = torch.argmax(masks_pred, dim=1)

        # Create visualization
        for idx in range(n_images):
            # Convert tensors to numpy arrays
            image = get_numpy_from_tensor(images[idx])
            pred = get_numpy_from_tensor(pred_labels[idx])
            true_mask = get_numpy_from_tensor(labels[idx])

            # Log to wandb
            wandb.log({
                f"predictions/{idx}": wandb.Image(
                    image,
                    masks={
                        "predictions": {"mask_data": pred, "class_labels": {i: str(i) for i in range(self.class_num)}},
                        "ground_truth": {"mask_data": true_mask, "class_labels": {i: str(i) for i in range(self.class_num)}}
                    }
                )
            }, step=step)

    def train(self, cfg):
        # Initialize wandb
        self.init_wandb(cfg)
        
        # initial identify
        train_meter = Average_Meter(list(self.losses.keys()) + ['total_loss'])
        train_iterator = Iterator(self.train_loader)
        best_valid_mIoU = -1
        model_path = "{cfg.model_folder}/{cfg.experiment_name}/model.pth".format(cfg=cfg)
        log_path = "{cfg.log_folder}/{cfg.experiment_name}/log_file.txt".format(cfg=cfg)
        check_folder(model_path)
        check_folder(log_path)

        # train
        for iteration in range(cfg.max_iter):
            images, labels = train_iterator.get()
            images, labels = images.cuda(), labels.cuda().long()
            # Use autocast for mixed precision training
            with autocast(device_type='cuda'):
                masks_pred, iou_pred = self.model(images)
                masks_pred = F.interpolate(masks_pred, self.original_size, mode="bilinear", align_corners=False)
                print(f"masks_pred.shape: {masks_pred.shape}  and  iou_pred.shape: {iou_pred.shape}")
                total_loss = torch.zeros(1).cuda()
                loss_dict = {}
                self._compute_loss(total_loss, loss_dict, masks_pred, labels, cfg)

            # Gradient scaling for mixed precision
            self.optimizer.zero_grad()
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            self.scheduler.step()
            loss_dict['total_loss'] = total_loss.item()
            train_meter.add(loss_dict)

            # Log training metrics to wandb
            if (iteration + 1) % cfg.log_iter == 0:
                metrics = train_meter.get(clear=True)
                self.log_metrics(metrics, iteration, "train")
                # Log sample predictions
                # self.log_images(images, masks_pred, labels, iteration)
                
                write_log(iteration=iteration, log_path=log_path, log_data=metrics,
                         status=self.exist_status[0],
                         writer=None, timer=self.train_timer)

            # eval
            if (iteration + 1) % cfg.eval_iter == 0:
                mIoU, _ = self._eval()
                if best_valid_mIoU == -1 or best_valid_mIoU < mIoU:
                    best_valid_mIoU = mIoU
                    save_model(self.model, model_path, parallel=self.the_number_of_gpu > 1)
                    print_and_save_log("saved model in {model_path}".format(model_path=model_path), path=log_path)
                
                # Log validation metrics
                val_metrics = {
                    'mIoU': mIoU, 
                    'best_valid_mIoU': best_valid_mIoU
                }
               
                self.log_metrics(val_metrics, iteration, "val")
                
                write_log(iteration=iteration, log_path=log_path, log_data=val_metrics,
                         status=self.exist_status[1],
                         writer=None, timer=self.eval_timer)

        # final process
        save_model(self.model, model_path, is_final=True, parallel=self.the_number_of_gpu > 1)
        wandb.finish()  # Close wandb run

    def test(self):
        pass

    def _eval(self):
        self.model.eval()
        self.eval_timer.start()
        class_names = self.val_loader.dataset.class_names
        eval_metric = mIoUOnline(class_names=class_names)
        
        with torch.no_grad():
            for index, (images, labels) in enumerate(self.val_loader):
                images = images.cuda()
                labels = labels.cuda()
                # Use autocast for evaluation as well
                with autocast(device_type='cuda'):
                    masks_pred, iou_pred = self.model(images)
                predictions = torch.argmax(masks_pred, dim=1)
                
                for batch_index in range(images.size()[0]):
                    pred_mask = get_numpy_from_tensor(predictions[batch_index])
                    gt_mask = get_numpy_from_tensor(labels[batch_index])
                    h, w = pred_mask.shape
                    gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    eval_metric.add(pred_mask, gt_mask)
        
        self.model.train()
        mean_iou, mean_iou_foreground = eval_metric.get(clear=True)
        return mean_iou, mean_iou_foreground

    def _compute_loss(self, total_loss, loss_dict, mask_pred, labels, cfg):
        """
        Due to the inputs of losses are different, so if you want to add new losses,
        you may need to modify the process in this function
        """
        loss_cfg = cfg.losses
        for index, item in enumerate(self.losses.items()):
            # item -> (key: loss_name, val: loss)
            real_labels = labels
            if loss_cfg[item[0]].label_one_hot:
                class_num = cfg.model.params.class_num
                real_labels = one_hot_embedding_3d(real_labels, class_num=class_num)
            
            # Compute loss
            tmp_loss = item[1](mask_pred, real_labels)
            
            # Ensure the loss is a scalar
            if isinstance(tmp_loss, torch.Tensor):
                if tmp_loss.numel() > 1:
                    tmp_loss = tmp_loss.mean()
                loss_val = tmp_loss.item()
            else:
                loss_val = float(tmp_loss)
                
            loss_dict[item[0]] = loss_val
            total_loss += loss_cfg[item[0]].weight * tmp_loss