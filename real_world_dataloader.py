import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class RealWorldMetalDataset(Dataset):
    """
    针对真实高反光金属工件的数据集类。
    支持：自动缩放、CLAHE增强、边缘生成。
    """
    def __init__(self, image_dir, mask_dir, target_size=(512, 512), transform=None, use_clahe=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.target_size = target_size
        self.transform = transform
        self.use_clahe = use_clahe
        
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace('.jpg', '.png').replace('.jpeg', '.png'))

        # 读取图像和掩膜
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 掩膜处理 (确保是单通道二值图)
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            # 如果没有掩膜，创建一个全黑的（用于推理或未标注数据测试）
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # 1. 预处理：针对高反光的 CLAHE 增强
        if self.use_clahe:
            # 转换为 LAB 空间进行亮度均衡，避免色彩失真
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            l = self.clahe.apply(l)
            lab = cv2.merge((l, a, b))
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # 2. 缩放到目标尺寸
        image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
        
        # 3. 生成边缘图 (Edge Mask)
        edge_mask = cv2.Canny(mask, 100, 200)
        edge_mask = (edge_mask > 0).astype(np.uint8) * 255

        # 4. 归一化与转换为 Tensor
        image_tensor = transforms.ToTensor()(image) # [0, 1]
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float() / 255.0 # [0, 1]
        edge_tensor = torch.from_numpy(edge_mask).unsqueeze(0).float() / 255.0 # [0, 1]

        if self.transform:
            # 如果有额外的数据增强（如旋转、翻转等）
            # 注意：对于分割任务，image 和 mask 必须同步增强
            pass 

        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'edge_mask': edge_tensor,
            'name': img_name
        }

class MixedMetalDataset(Dataset):
    """
    混合数据集：将合成数据和真实数据按比例混合。
    用于解决真实数据量不足时的过拟合问题。
    """
    def __init__(self, synthetic_dataset, real_dataset, real_ratio=0.5):
        self.synthetic_dataset = synthetic_dataset
        self.real_dataset = real_dataset
        self.real_ratio = real_ratio
        
        # 总长度以真实数据为基准进行缩放，或者取两者之和
        self.length = max(len(synthetic_dataset), len(real_dataset))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 按概率选择从哪个数据集采样
        if np.random.rand() < self.real_ratio and len(self.real_dataset) > 0:
            real_idx = idx % len(self.real_dataset)
            return self.real_dataset[real_idx]
        else:
            syn_idx = idx % len(self.synthetic_dataset)
            return self.synthetic_dataset[syn_idx]

def get_mixed_dataloader(synthetic_ds, real_img_dir, real_mask_dir, batch_size=4, real_ratio=0.5):
    """
    便捷函数：获取混合数据加载器。
    """
    real_ds = RealWorldMetalDataset(real_img_dir, real_mask_dir)
    mixed_ds = MixedMetalDataset(synthetic_ds, real_ds, real_ratio=real_ratio)
    
    return DataLoader(mixed_ds, batch_size=batch_size, shuffle=True, num_workers=4)

if __name__ == '__main__':
    # 简单测试
    # dataset = RealWorldMetalDataset('path/to/images', 'path/to/masks')
    # loader = DataLoader(dataset, batch_size=2)
    print("Real-world dataloader module initialized.")
