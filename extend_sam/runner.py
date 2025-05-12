from datasets import Iterator
from .utils import Average_Meter, Timer, print_and_save_log, mIoUOnline, get_numpy_from_tensor, save_model, write_log, \
    check_folder, one_hot_embedding_3d, apply_label_colors, overlay_mask_on_image, create_visualization
import torch
import cv2
import torch.nn.functional as F
import os
import torch.nn as nn
from torch.amp import autocast, GradScaler
import wandb
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image


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
        self.class_num = None

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
        
        n_images = min(images.shape[0], max_images)
        images = images[:n_images]
        masks_pred = masks_pred[:n_images]
        labels = labels[:n_images]
        pred_labels = torch.argmax(masks_pred, dim=1)

        for idx in range(n_images):
            image = get_numpy_from_tensor(images[idx])
            pred = get_numpy_from_tensor(pred_labels[idx])
            true_mask = get_numpy_from_tensor(labels[idx])

            # Convert image from [C, H, W] to [H, W, C]
            if image.ndim == 3 and image.shape[0] in [1, 3]:
                image = image.transpose(1, 2, 0)
            image = (image * 255).astype(np.uint8)

            # Resize all to match image size
            h, w = image.shape[:2]
            pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
            true_mask = cv2.resize(true_mask, (w, h), interpolation=cv2.INTER_NEAREST)

            # Color scheme (same as test)
            colors = {
                0: [0, 0, 0],        # Background - Black
                1: [0, 255, 0],      # Room -> Green
                2: [255, 0, 0],      # Wall -> Red
                3: [0, 0, 255],      # Door -> Blue
                4: [255, 255, 0],    # Window -> Yellow
            }
            pred_colored = np.zeros((h, w, 3), dtype=np.uint8)
            gt_colored = np.zeros((h, w, 3), dtype=np.uint8)
            for class_idx, color in colors.items():
                pred_colored[pred == class_idx] = color
                gt_colored[true_mask == class_idx] = color

            # Side-by-side visualization
            vis_image = np.concatenate([image, pred_colored, gt_colored], axis=1)

            # Log to wandb
            wandb.log({
                f"visualization/{idx}": wandb.Image(vis_image, caption="[Input | Prediction | Ground Truth]")
            }, step=step)

    def train(self, cfg):
        # Initialize wandb
        self.init_wandb(cfg)
        self.class_num = cfg.model.params.class_num
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
                print(f"masks_pred.shape: {masks_pred.shape}  and  labels.shape: {labels.shape}")
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
                self.log_images(images, masks_pred, labels, iteration)
                
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

    def test(self, cfg):
        """Test function that performs evaluation and visualization"""
        self.model.eval()
        self.eval_timer.start()
        class_names = self.val_loader.dataset.class_names
        eval_metric = mIoUOnline(class_names=class_names)
        
        # Create output directory for visualizations if it doesn't exist
        vis_dir = os.path.join(cfg.model_folder, cfg.experiment_name, 'test_visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        with torch.no_grad():
            for index, (images, labels) in enumerate(self.val_loader):
                images = images.cuda()
                labels = labels.cuda()
                
                # Use autocast for evaluation
                with autocast(device_type='cuda'):
                    masks_pred, iou_pred = self.model(images)
                predictions = torch.argmax(masks_pred, dim=1)
                
                for batch_index in range(images.size()[0]):
                    # Get predictions and ground truth
                    pred_mask = get_numpy_from_tensor(predictions[batch_index])
                    gt_mask = get_numpy_from_tensor(labels[batch_index])
                    image = get_numpy_from_tensor(images[batch_index])
                    
                    # Convert image from tensor format (C,H,W) to numpy format (H,W,C)
                    image = image.transpose(1, 2, 0)
                    # Denormalize image if needed (assuming normalization was applied during training)
                    image = (image * 255).astype(np.uint8)
                    
                    # Resize all images and masks to the same size (using original image size)
                    h, w = image.shape[:2]
                    pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    
                    # Create colored masks
                    pred_colored = np.zeros((h, w, 3), dtype=np.uint8)
                    gt_colored = np.zeros((h, w, 3), dtype=np.uint8)
                    
                    # Assign colors to each class (you can customize these colors)
                    colors = {
                        0: [0, 0, 0],        # Background - Black
                        1: [0, 255, 0],      # Room -> Green
                        2: [255, 0, 0],      # Wall -> Red
                        3: [0, 0, 255],      # Door -> Blue
                        4: [255, 255, 0],    # Window -> Yellow
                    }
                    
                    # Color the masks
                    for class_idx, color in colors.items():
                        pred_colored[pred_mask == class_idx] = color
                        gt_colored[gt_mask == class_idx] = color
                    
                    # Create visualization
                    vis_image = np.concatenate([image, pred_colored, gt_colored], axis=1)
                    
                    # Add to evaluation metric
                    eval_metric.add(pred_mask, gt_mask)
                    
                    # Save visualization
                    vis_path = os.path.join(vis_dir, f'sample_{index}_{batch_index}.png')
                    cv2.imwrite(vis_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        
        # Get evaluation metrics
        mean_iou, mean_iou_foreground = eval_metric.get(clear=True)
        
        # Print results
        print(f"\nTest Results:")
        print(f"mIoU: {mean_iou:.4f}")
        print(f"mIoU (foreground): {mean_iou_foreground:.4f}")
        print(f"Visualizations saved to: {vis_dir}")
        
        return mean_iou, mean_iou_foreground

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
                    # print(f"pred_mask.shape: {pred_mask.shape}  and  gt_mask.shape: {gt_mask.shape}")
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

    

    def infer(self, image_dir, output_dir, image_size=(1024, 1024)):
        """Run inference on a directory of images without ground truth masks.
        
        Args:
            image_dir (str): Directory containing input images
            output_dir (str): Directory to save visualization results
            image_size (tuple): Size to resize input images to (height, width)
        """
        self.model.eval()
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'visualization'), exist_ok=True)
        
        # Get list of image files
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        with torch.no_grad():
            for img_file in tqdm(image_files, desc="Processing images"):
                # Load and preprocess image
                img_path = os.path.join(image_dir, img_file)
                orig_image = cv2.imread(img_path)
                orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
                
                # Store original size for later
                orig_h, orig_w = orig_image.shape[:2]
                
                # Resize and normalize image for model input
                image = cv2.resize(orig_image, (image_size[1], image_size[0]))
                image = image.astype(np.float32) / 255.0
                image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
                
                # Move to GPU and normalize
                image = image.cuda()
                
                # Run inference
                masks_pred, _ = self.model(image)
                
                # Get predictions and resize back to original size
                masks_pred = F.interpolate(masks_pred, (orig_h, orig_w), mode="bilinear", align_corners=False)
                predictions = torch.argmax(masks_pred, dim=1)[0]
                pred_mask = get_numpy_from_tensor(predictions)
                pred_mask[pred_mask == 1] = 0
                
                # Create visualization
                base_name = os.path.splitext(img_file)[0]
                create_visualization(
                    orig_image, 
                    pred_mask,
                    os.path.join(output_dir, 'visualization', f"{base_name}_comparison.png")
                )
                
                # Save individual mask and overlay images
                # colored_mask = apply_label_colors(pred_mask)
                # overlay_pred = overlay_mask_on_image(orig_image, pred_mask)
                
                # cv2.imwrite(
                #     os.path.join(output_dir, f"{base_name}_mask.png"), 
                #     cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR)
                # )
                # cv2.imwrite(
                #     os.path.join(output_dir, f"{base_name}_overlay.png"),
                #     cv2.cvtColor(overlay_pred, cv2.COLOR_RGB2BGR)
                # )