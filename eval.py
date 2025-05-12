'''
@copyright ziqi-jin
'''
import argparse
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from datasets import get_dataset
from extend_sam import get_model, get_runner

supported_tasks = ['detection', 'semantic_seg', 'instance_seg']
parser = argparse.ArgumentParser()
parser.add_argument('--task_name', default='semantic_seg', type=str)
parser.add_argument('--cfg', default=None, type=str)
parser.add_argument('--model_path', required=True, type=str, help='Path to the trained model checkpoint')
parser.add_argument('--mode', required=True, choices=['test', 'infer'], default='test', type=str, help='Determine the mode of evaluation')


if __name__ == '__main__':
    args = parser.parse_args()
    task_name = args.task_name
    if args.cfg is not None:
        config = OmegaConf.load(args.cfg)
    else:
        assert task_name in supported_tasks, "Please input the supported task name."
        config = OmegaConf.load("./config/{task_name}.yaml".format(task_name=args.task_name))

    val_cfg = config.val
    test_cfg = config.test
    infer_cfg = config.infer
    # Load model
    model = get_model(model_name=config.train.model.sam_name, **config.train.model.params)
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path)
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    # Run evaluation
    if args.mode == 'test':
        # Set up validation dataset and dataloader
        val_dataset = get_dataset(test_cfg.dataset)
        val_loader = DataLoader(val_dataset, batch_size=test_cfg.bs, shuffle=False, 
                            num_workers=test_cfg.num_workers, drop_last=test_cfg.drop_last)
        # Initialize runner with None for training-related components since we're only evaluating
        runner = get_runner(config.train.runner_name)(model, None, None, None, val_loader, None)
    

        print("Starting evaluation...")
        mIoU, mIoU_foreground = runner.test(test_cfg)
        print(f"Evaluation Results:")
        print(f"mIoU: {mIoU:.4f}")
        print(f"mIoU (foreground): {mIoU_foreground:.4f}") 
    elif args.mode == 'infer':
        runner = get_runner(config.train.runner_name)(model, None, None, None, None, None)
        runner.infer(infer_cfg.infer.image_dir, infer_cfg.infer.output_dir)