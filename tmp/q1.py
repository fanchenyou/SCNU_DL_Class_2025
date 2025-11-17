import cv2
import numpy as np
import matplotlib.pyplot as plt

# 将彩图转为单通道灰度图
img = cv2.imread("jay.png")  
print("原始彩图尺寸:", img.shape) 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("原始灰度图尺寸:", gray.shape) 

# 定义5个Harr滤波器（尺寸5x5~10x10）
harr_filters = [
    # 1. 5×5 水平边缘
    np.array([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0],
        [-1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1]
    ]),
    # 2. 5×5 垂直边缘
    np.array([
        [1, 1, 0, -1, -1],
        [1, 1, 0, -1, -1],
        [1, 1, 0, -1, -1],
        [1, 1, 0, -1, -1],
        [1, 1, 0, -1, -1]
    ]),
    # 3. 7×7 垂直线条
    np.array([
        [1, 1, -2, -2, -2, 1, 1],
        [1, 1, -2, -2, -2, 1, 1],
        [1, 1, -2, -2, -2, 1, 1],
        [1, 1, -2, -2, -2, 1, 1],
        [1, 1, -2, -2, -2, 1, 1],
        [1, 1, -2, -2, -2, 1, 1],
        [1, 1, -2, -2, -2, 1, 1]
    ]),
    # 4. 7×7 水平线条
    np.array([
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [-2, -2, -2, -2, -2, -2, -2],
        [-2, -2, -2, -2, -2, -2, -2],
        [-2, -2, -2, -2, -2, -2, -2],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],

    ]),
    # 5. 8×8 对角特征
    np.array([
        [-1, -1, -1, -1, 1, 1, 1, 1],
        [-1, -1, -1, -1, 1, 1, 1, 1],
        [-1, -1, -1, -1, 1, 1, 1, 1],
        [-1, -1, -1, -1, 1, 1, 1, 1],
        [1, 1, 1, 1, -1, -1, -1, -1],
        [1, 1, 1, 1, -1, -1, -1, -1],
        [1, 1, 1, 1, -1, -1, -1, -1],
        [1, 1, 1, 1, -1, -1, -1, -1]
    ])
]

# 手动卷积函数
def manual_convolution(image, filter):
    img_h, img_w = image.shape
    filter_h, filter_w = filter.shape
    out_h = img_h - filter_h + 1
    out_w = img_w - filter_w + 1
    feature_map = np.zeros((out_h, out_w), dtype=np.float32)
    
    for i in range(out_h):
        for j in range(out_w):
            img_patch = image[i:i+filter_h, j:j+filter_w]
            feature_map[i, j] = np.sum(img_patch * filter)
    return feature_map

# 生成并保存所有特征图
for i, filt in enumerate(harr_filters):
    feature = manual_convolution(gray, filt)
    # 归一化时判断，避免除以0
    if feature.max() - feature.min() == 0:
        feature_norm = np.zeros_like(feature, dtype=np.uint8)
    else:
        feature_norm = ((feature - feature.min()) / (feature.max() - feature.min()) * 255).astype(np.uint8)

    # 可视化
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('Original Gray Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(feature_norm, cmap='gray')
    plt.title(f'Harr Filter {i+1} Feature Map')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    cv2.imwrite(f'harr_feature_{i+1}.jpg', feature_norm)
    
cv2.imwrite(f'original_gray.jpg', gray)