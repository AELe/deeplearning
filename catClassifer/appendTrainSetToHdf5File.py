import h5py
import numpy as np
from PIL import Image
import os

# 文件夹中的新JPEG图片和对应标签
new_image_folder = "new_images"  # 假设新图片都在这个文件夹中
image_label_dict = {
    "/home/zsy/Desktop/test/e.jpg": 1, 
    "/home/zsy/Desktop/test/h.jpg": 1,
    "/home/zsy/Desktop/test/f.jpg": 1,
    "/home/zsy/Desktop/test/j.jpg": 1,
    # 添加更多的图片文件名和对应标签
}

# 打开现有的HDF5文件以读取数据
with h5py.File('/home/zsy/Workspace/deepLearning/homework/deep-learning/datasets/train_catvnoncat.h5', 'r') as file:
    existing_train_set_x = file['train_set_x']
    existing_train_set_y = file['train_set_y']
    existing_list_classes = file['list_classes']

    # 获取现有数据集的形状
    existing_size = existing_train_set_x.shape[0]
    new_size = existing_size + len(image_label_dict)  # 新数据的数量

    # 创建一个新的HDF5文件
    with h5py.File('/home/zsy/Workspace/deepLearning/homework/deep-learning/datasets/train_catvnoncat_updated.h5', 'w') as new_file:
        # 创建新的数据集并启用chunking
        new_train_set_x = new_file.create_dataset(
            'train_set_x', (new_size, 64, 64, 3), chunks=(1, 64, 64, 3), maxshape=(None, 64, 64, 3))
        new_train_set_y = new_file.create_dataset(
            'train_set_y', (new_size,), chunks=(1,), maxshape=(None,))
        new_list_classes = new_file.create_dataset(
            'list_classes', data=existing_list_classes[:])

        # 复制现有数据到新的数据集
        new_train_set_x[:existing_size] = existing_train_set_x[:]
        new_train_set_y[:existing_size] = existing_train_set_y[:]

        # 添加新的图片和标签
        for filename, label in image_label_dict.items():
            img_path = os.path.join(new_image_folder, filename)
            img = Image.open(img_path)
            img = img.resize((64, 64))  # 调整图片大小为64x64
            img_array = np.array(img)
            new_train_set_x[existing_size] = img_array
            new_train_set_y[existing_size] = label
            existing_size += 1

print("新的图片数据已成功添加到训练集中！")


