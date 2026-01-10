import cv2
import numpy as np

def exposure_fusion(images):
    """
    使用 Mertens 算法进行多重曝光融合。
    该算法不需要曝光时间信息，直接根据对比度、饱和度和曝光适度进行融合。
    """
    merge_mertens = cv2.createMergeMertens()
    res_mertens = merge_mertens.process(images)
    
    # 将结果转换为 8 位图像
    res_mertens_8bit = np.clip(res_mertens * 255, 0, 255).astype('uint8')
    return res_mertens_8bit

def adaptive_image_enhancement(image):
    """
    自适应图像增强：CLAHE + 引导滤波。
    用于在抑制噪声的同时增强边缘。
    """
    # 1. CLAHE (对比度受限的自适应直方图均衡化)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # 2. 引导滤波 (Guided Filter) - 保持边缘的同时平滑图像
    # 这里使用 OpenCV 的 ximgproc 模块，如果未安装，可以使用双边滤波替代
    try:
        from cv2.ximgproc import guidedFilter
        enhanced_img = guidedFilter(guide=enhanced_img, src=enhanced_img, radius=10, eps=100)
    except ImportError:
        # 备选方案：双边滤波
        enhanced_img = cv2.bilateralFilter(enhanced_img, d=9, sigmaColor=75, sigmaSpace=75)
        
    return enhanced_img

def simulate_polarization_effect(images):
    """
    模拟偏振效果：通过分析多张图像的最小像素值来抑制镜面反射。
    原理：镜面反射在不同光照/角度下变化剧烈，而漫反射相对稳定。
    """
    # 将图像转换为浮点型
    imgs_float = [img.astype(np.float32) for img in images]
    
    # 取所有图像对应像素的最小值，这有助于消除随机出现的强反光斑
    min_img = np.minimum.reduce(imgs_float)
    
    return min_img.astype(np.uint8)

if __name__ == "__main__":
    # 示例用法说明
    print("HDR Processing Module Loaded.")
    # 假设有三张不同曝光的图像：img_under, img_normal, img_over
    # images = [img_under, img_normal, img_over]
    # hdr_res = exposure_fusion(images)
    # final_res = adaptive_image_enhancement(hdr_res)
