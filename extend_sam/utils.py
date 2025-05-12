'''
@copyright ziqi-jin
'''
import time
import numpy as np
import torch
import torch.nn.functional as F
import os.path as osp
import os


def fix_params(model):
    for name, param in model.named_parameters():
        param.requires_grad = False


def load_params(model, params):
    pass


def get_opt_pamams(model, lr_list, group_keys, wd_list):
    '''

    :param model: model
    :param lr_list: list, contain the lr for each params group
    :param wd_list: list, contain the weight decay for each params group
    :param group_keys: list of list, according to the sub list to divide params to different groups
    :return: list of dict
    '''
    assert len(lr_list) == len(group_keys), "lr_list should has the same length as group_keys"
    assert len(lr_list) == len(wd_list), "lr_list should has the same length as wd_list"
    params_group = [[] for _ in range(len(lr_list))]
    for name, value in model.named_parameters():
        for index, g_keys in enumerate(group_keys):
            for g_key in g_keys:
                if g_key in name:
                    params_group[index].append(value)
    return [{'params': params_group[i], 'lr': lr_list[i], 'weight_decay': wd_list[i]} for i in range(len(lr_list))]


class Timer:

    def __init__(self):
        self.start_time = 0.0
        self.end_time = 0.0

        self.start()

    def start(self):
        self.start_time = time.time()

    def end(self, ms=False, clear=False):
        self.end_time = time.time()

        if ms:
            duration = int((self.end_time - self.start_time) * 1000)
        else:
            duration = int(self.end_time - self.start_time)

        if clear:
            self.start()

        return duration


class Average_Meter:
    def __init__(self, keys):
        self.keys = keys
        self.clear()

    def add(self, dic):
        for key, value in dic.items():
            self.data_dic[key].append(value)

    def get(self, keys=None, clear=False):
        if keys is None:
            keys = self.keys

        dataset = {}
        for key in keys:
            dataset[key] = float(np.mean(self.data_dic[key]))

        if clear:
            self.clear()

        return dataset

    def clear(self):
        self.data_dic = {key: [] for key in self.keys}


def print_and_save_log(message, path):
    print(message)

    with open(path, 'a+') as f:
        f.write(message + '\n')


class mIoUOnline:
    def __init__(self, class_names):
        self.class_names = ['background'] + class_names
        self.class_num = len(self.class_names)
        self.clear()

    def get_data(self, pred_mask, gt_mask):
        # Only consider non-background pixels
        obj_mask = (gt_mask < 255) & (gt_mask > 0)  # Exclude background (0) and invalid (255)
        correct_mask = (pred_mask == gt_mask) * obj_mask

        P_list, T_list, TP_list = [], [], []
        for i in range(self.class_num):
            P_list.append(np.sum((pred_mask == i) * obj_mask))
            T_list.append(np.sum((gt_mask == i) * obj_mask))
            TP_list.append(np.sum((gt_mask == i) * correct_mask))

        return (P_list, T_list, TP_list)

    def add_using_data(self, data):
        P_list, T_list, TP_list = data
        for i in range(self.class_num):
            self.P[i] += P_list[i]
            self.T[i] += T_list[i]
            self.TP[i] += TP_list[i]

    def add(self, pred_mask, gt_mask):
        # Only consider non-background pixels
        obj_mask = (gt_mask < 255) & (gt_mask > 0)  # Exclude background (0) and invalid (255)
        correct_mask = (pred_mask == gt_mask) * obj_mask

        for i in range(self.class_num):
            self.P[i] += np.sum((pred_mask == i) * obj_mask)
            self.T[i] += np.sum((gt_mask == i) * obj_mask)
            self.TP[i] += np.sum((gt_mask == i) * correct_mask)

    def get(self, detail=False, clear=True):
        IoU_dic = {}
        IoU_list = []
        FP_list = []  # over activation
        FN_list = []  # under activation

        # Calculate IoU for all classes but only include non-background in metrics
        for i in range(self.class_num):
            IoU = self.TP[i] / (self.T[i] + self.P[i] - self.TP[i] + 1e-10) * 100
            FP = (self.P[i] - self.TP[i]) / (self.T[i] + self.P[i] - self.TP[i] + 1e-10)
            FN = (self.T[i] - self.TP[i]) / (self.T[i] + self.P[i] - self.TP[i] + 1e-10)

            IoU_dic[self.class_names[i]] = IoU

            # Only include non-background classes in the lists
            if i > 0:  # Skip background class
                IoU_list.append(IoU)
                FP_list.append(FP)
                FN_list.append(FN)

        # Calculate metrics only on non-background classes
        mIoU = np.mean(np.asarray(IoU_list))  # Already only includes non-background
        mIoU_foreground = mIoU  # Same as mIoU since we're only considering foreground

        FP = np.mean(np.asarray(FP_list))
        FN = np.mean(np.asarray(FN_list))

        if clear:
            self.clear()

        if detail:
            return mIoU, mIoU_foreground, IoU_dic, FP, FN
        else:
            return mIoU, mIoU_foreground

    def clear(self):
        self.TP = []
        self.P = []
        self.T = []

        for _ in range(self.class_num):
            self.TP.append(0)
            self.P.append(0)
            self.T.append(0)


def get_numpy_from_tensor(tensor):
    return tensor.cpu().detach().numpy()


def save_model(model, model_path, parallel=False, is_final=False):
    if is_final:
        model_path_split = model_path.split('.')
        model_path = model_path_split[0] + "_final.pth"
    if parallel:
        torch.save(model.module.state_dict(), model_path)
    else:
        torch.save(model.state_dict(), model_path)


def write_log(iteration, log_path, log_data, status, writer, timer):
    log_data['iteration'] = iteration
    log_data['time'] = timer.end(clear=True)
    message = "iteration : {val}, ".format(val=log_data['iteration'])
    for key, value in log_data.items():
        if key == 'iteration':
            continue
        message += "{key} : {val}, ".format(key=key, val=value)
    message = message[:-2]  # + '\n'
    print_and_save_log(message, log_path)
    # visualize
    if writer is not None:
        for key, value in log_data.items():
            writer.add_scalar("{status}/{key}".format(status=status, key=key), value, iteration)


def check_folder(file_path, is_folder=False):
    '''

    :param file_path: the path of file, default input is a complete file name with dir path.
    :param is_folder: if the input is a dir, not a file_name, is_folder should be True
    :return: no return, this function will check and judge whether need to make dirs.
    '''
    if is_folder:
        if not osp.exists(is_folder):
            os.makedirs(file_path)

    else:
        splits = file_path.split("/")
        folder_name = "/".join(splits[:-1])
        if not osp.exists(folder_name):
            os.makedirs(folder_name)


def one_hot_embedding_3d(labels, class_num=21):
    '''

    :param real_labels: B H W
    :param class_num: N
    :return: B N H W
    '''
    one_hot_labels = labels.clone()
    one_hot_labels[one_hot_labels == 255] = 0 # 0 is background
    return F.one_hot(one_hot_labels, num_classes=class_num).permute(0, 3, 1, 2).contiguous().float()


# Constants for visualization
LABEL_COLORS = {
    0: [0, 0, 0],        # Background - Black
    1: [0, 255, 0],      # Room -> Green
    2: [255, 0, 0],      # Wall -> Red
    3: [0, 0, 255],      # Door -> Blue
    4: [255, 255, 0],    # Window -> Yellow
}

LABEL_NAMES = {
    0: 'Background',
    1: 'Room',
    2: 'Wall',
    3: 'Door',
    4: 'Window'
}

def apply_label_colors(pred_mask):
    """
    Converts a prediction mask to a colored mask based on predefined label colors.
    
    Args:
        pred_mask (numpy.ndarray): The prediction mask array
    
    Returns:
        numpy.ndarray: The colored mask
    """
    # Initialize a color image (3 channels: RGB)
    colored_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    
    # Apply the colors based on the labels in the mask
    for label, color in LABEL_COLORS.items():
        if label == 1: continue  # Skip room label
        colored_mask[pred_mask == label] = color
    
    return colored_mask

def overlay_mask_on_image(image, pred_mask):
    """
    Overlays the colored mask onto the original image.
    
    Args:
        image (numpy.ndarray): The original RGB image
        pred_mask (numpy.ndarray): The prediction mask array
    
    Returns:
        numpy.ndarray: The image with the overlayed mask
    """
    colored_mask = apply_label_colors(pred_mask)
    
    if image.shape[2] != 3:
        raise ValueError("The input image must have 3 channels (RGB).")
        
    overlayed_image = image.copy()
    
    # Create mask for valid regions (excluding background and room)
    mask_region = (pred_mask > 0) & (pred_mask < 5)
    
    # Apply mask overlay
    overlayed_image[mask_region] = colored_mask[mask_region]
    
    return overlayed_image

def create_visualization(image, pred_mask, output_path):
    """
    Creates and saves a side-by-side visualization of the original image and prediction.
    
    Args:
        image (numpy.ndarray): The original RGB image
        pred_mask (numpy.ndarray): The prediction mask array
        output_path (str): Path to save the visualization
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    
    # Set pred_mask class 1 (room) to 0 (background)
    pred_mask = pred_mask.copy()
    pred_mask[pred_mask == 1] = 0
    
    # Generate overlay
    overlay_pred = overlay_mask_on_image(image, pred_mask)
    
    # Create matplotlib figure
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot original image
    axs[0].imshow(image)
    axs[0].set_title("Input RGB")
    
    # Plot prediction overlay
    axs[1].imshow(overlay_pred)
    axs[1].set_title("Prediction")
    
    # Add legend
    legend_handles = [
        Patch(color=np.array(LABEL_COLORS[label])/255.0, label=LABEL_NAMES[label])
        for label in LABEL_COLORS
        if label != 1  # Skip room label
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=4, fontsize='small')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    
    # Save visualization
    plt.savefig(output_path, dpi=500)
    plt.close()
