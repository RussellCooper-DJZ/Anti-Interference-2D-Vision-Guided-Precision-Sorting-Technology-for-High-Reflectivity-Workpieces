import cv2
import numpy as np

def exposure_fusion(images):
    """
    使用 Mertens 算法进行多重曝光融合。
    该算法不需要曝光时间信息，直接根据对比度、饱和度和曝光适度进行融合。

    Performs multi-exposure fusion using the Mertens algorithm.
    This algorithm does not require exposure time information and directly fuses based on contrast, saturation, and exposure adequacy.

    Führt eine Mehrfachbelichtungsfusion mit dem Mertens-Algorithmus durch.
    Dieser Algorithmus benötigt keine Belichtungszeitinformationen und fusioniert direkt basierend auf Kontrast, Sättigung und Belichtungsangemessenheit.
    """
    merge_mertens = cv2.createMergeMertens()
    res_mertens = merge_mertens.process(images)
    
    # 将结果转换为 8 位图像
    # Convert the result to an 8-bit image
    # Konvertiert das Ergebnis in ein 8-Bit-Bild
    res_mertens_8bit = np.clip(res_mertens * 255, 0, 255).astype('uint8')
    return res_mertens_8bit

def adaptive_image_enhancement(image):
    """
    自适应图像增强：CLAHE + 引导滤波。
    用于在抑制噪声的同时增强边缘。

    Adaptive image enhancement: CLAHE + Guided Filter.
    Used to enhance edges while suppressing noise.

    Adaptive Bildverbesserung: CLAHE + Guided Filter.
    Wird verwendet, um Kanten zu verbessern und gleichzeitig Rauschen zu unterdrücken.
    """
    # 1. CLAHE (对比度受限的自适应直方图均衡化)
    # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # 1. CLAHE (Kontrastbegrenzte adaptive Histogramm-Entzerrung)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # 2. 引导滤波 (Guided Filter) - 保持边缘的同时平滑图像
    # 这里使用 OpenCV 的 ximgproc 模块，如果未安装，可以使用双边滤波替代
    # 2. Guided Filter - Smooths the image while preserving edges.
    # Here, OpenCV's ximgproc module is used; if not installed, bilateral filtering can be used as an alternative.
    # 2. Guided Filter - Glättet das Bild unter Beibehaltung der Kanten.
    # Hier wird das ximgproc-Modul von OpenCV verwendet; falls nicht installiert, kann die bilaterale Filterung als Alternative verwendet werden.
    try:
        from cv2.ximgproc import guidedFilter
        enhanced_img = guidedFilter(guide=enhanced_img, src=enhanced_img, radius=10, eps=100)
    except ImportError:
        # 备选方案：双边滤波
        # Alternative: Bilateral filtering
        # Alternative: Bilaterale Filterung
        enhanced_img = cv2.bilateralFilter(enhanced_img, d=9, sigmaColor=75, sigmaSpace=75)
        
    return enhanced_img

def simulate_polarization_effect(images):
    """
    模拟偏振效果：通过分析多张图像的最小像素值来抑制镜面反射。
    原理：镜面反射在不同光照/角度下变化剧烈，而漫反射相对稳定。

    Simulates polarization effect: Suppresses specular reflection by analyzing the minimum pixel values of multiple images.
    Principle: Specular reflection varies greatly under different lighting/angles, while diffuse reflection is relatively stable.

    Simuliert den Polarisationseffekt: Unterdrückt spiegelnde Reflexionen durch Analyse der minimalen Pixelwerte mehrerer Bilder.
    Prinzip: Spiegelnde Reflexionen variieren stark unter verschiedenen Beleuchtungen/Winkeln, während diffuse Reflexionen relativ stabil sind.
    """
    # 将图像转换为浮点型
    # Convert images to float type
    # Konvertiert Bilder in den Gleitkommatyp
    imgs_float = [img.astype(np.float32) for img in images]
    
    # 取所有图像对应像素的最小值，这有助于消除随机出现的强反光斑
    # Take the minimum pixel value across all images, which helps eliminate randomly appearing strong glare spots.
    # Nimmt den minimalen Pixelwert über alle Bilder hinweg, was hilft, zufällig auftretende starke Blendflecken zu eliminieren.
    min_img = np.minimum.reduce(imgs_float)
    
    return min_img.astype(np.uint8)

if __name__ == "__main__":
    # 示例用法说明
    # Example usage instructions
    # Beispiel für die Verwendung
    print("HDR Processing Module Loaded.")
    # 假设有三张不同曝光的图像：img_under, img_normal, img_over
    # Assume there are three images with different exposures: img_under, img_normal, img_over
    # Angenommen, es gibt drei Bilder mit unterschiedlichen Belichtungen: img_under, img_normal, img_over
    # images = [img_under, img_normal, img_over]
    # hdr_res = exposure_fusion(images)
    # final_res = adaptive_image_enhancement(hdr_res)
