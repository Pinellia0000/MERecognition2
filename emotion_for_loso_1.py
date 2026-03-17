import os
import shutil
import pandas as pd
from tqdm import tqdm
import zipfile
import datetime


def zip_frames(packagePath, zipPath):
    """
    packagePath: 文件夹路径
    zipPath: 压缩包路径
    """
    zip = zipfile.ZipFile(zipPath, 'w', zipfile.ZIP_DEFLATED)
    for path, dirNames, fileNames in os.walk(packagePath):
        fpath = path.replace(packagePath, '')
        for name in fileNames:
            fullName = os.path.join(path, name)
            name = fpath + '\\' + name
            zip.write(fullName, name)
    zip.close()


def print_directory_structure(root_dir, indent=""):
    """
    递归打印目录结构
    """
    # 获取当前目录下的所有文件和文件夹，并排序（保证输出稳定）
    items = sorted(os.listdir(root_dir))

    for idx, item in enumerate(items):
        path = os.path.join(root_dir, item)
        # 判断是否是最后一个元素
        pointer = "└── " if idx == len(items) - 1 else "├── "
        print(indent + pointer + item)

        if os.path.isdir(path):
            # 如果是文件夹，递归打印子目录
            extension = "    " if idx == len(items) - 1 else "│   "
            print_directory_structure(path, indent + extension)


def CASME2_5c_3c(CASME2_onset_apex_offset_retinaface, CASME2_optflow_retinaface, data_folder_5, data_folder_3, annotation_file):
    """
    CASME2
    5类：'happiness': 0,
        'surprise': 1,
        'disgust': 2,
        'repression': 3,
        'others': 4
    3类:'positive': 0,
        'negative': 1,
        'surprise': 2,
    注意：positive包括happiness  negative包括repression和disgust  surprise包括surprise
    先进行5分类
    然后将5分类复制进3分类
    """
    # 情绪映射字典
    label_dict = {
        'happiness': 0,
        'surprise': 1,
        'disgust': 2,
        'repression': 3,
        'others': 4
    }

    # 创建情绪文件夹 0~4 5分类
    for label in label_dict.values():
        os.makedirs(os.path.join(data_folder_5, str(label)), exist_ok=True)

    # 创建情绪文件夹 0~2 3分类
    for label in label_dict.values():
        os.makedirs(os.path.join(data_folder_3, str(label)), exist_ok=True)

    # 读取注释文件
    anno_df = pd.read_excel(annotation_file)
    # 找出该被试的注释行
    # 第一层目录名称的最后两位对应注释文件中的Subject
    # 图片名称中如果包含对应注释文件中的Filename
    # 根据这两个将对应行确定
    # 将获取该行的Estimated Emotion字段值
    # 根据获取的字段值将对应目录下的所有图片复制到字段字典对应的文件夹下
    # 未防止重复，复制的图片名称前增加原始第一层目录的名称，按下划线连接
    # 遍历被试
    for sub_num in tqdm(range(1, 27), desc="Processing subjects"):
        sub_prefix = f'sub{sub_num:02d}'
        sub_folder_path = os.path.join(CASME2_onset_apex_offset_retinaface, sub_prefix)
        if not os.path.exists(sub_folder_path):
            continue

        # 找出该被试的注释行
        sub_df = anno_df[anno_df['Subject'].apply(lambda x: f'sub{x:02d}') == sub_prefix]

        # 遍历该被试目录下的每张图片
        for img_name in os.listdir(sub_folder_path):
            img_path = os.path.join(sub_folder_path, img_name)

            # 匹配注释文件中的 Filename
            matched_rows = sub_df[sub_df['Filename'].apply(lambda x: img_name.startswith(x))]
            if matched_rows.empty:
                continue

            for _, row in matched_rows.iterrows():
                emotion = row['Estimated Emotion']
                if emotion not in label_dict:
                    continue
                label_id = label_dict[emotion]

                dst_dir = os.path.join(data_folder_5, str(label_id))
                os.makedirs(dst_dir, exist_ok=True)

                new_name = f"{sub_prefix}_{img_name}"
                dst_path = os.path.join(dst_dir, new_name)
                shutil.copy(img_path, dst_path)

        # 光流目录同理
        optflow_sub_folder = os.path.join(CASME2_optflow_retinaface, sub_prefix)
        if os.path.exists(optflow_sub_folder):
            for img_name in os.listdir(optflow_sub_folder):
                img_path = os.path.join(optflow_sub_folder, img_name)
                matched_rows = sub_df[sub_df['Filename'].apply(lambda x: img_name.startswith(x))]
                if matched_rows.empty:
                    continue

                for _, row in matched_rows.iterrows():
                    emotion = row['Estimated Emotion']
                    if emotion not in label_dict:
                        continue
                    label_id = label_dict[emotion]
                    dst_dir = os.path.join(data_folder_5, str(label_id))
                    os.makedirs(dst_dir, exist_ok=True)

                    new_name = f"{sub_prefix}_{img_name}"
                    dst_path = os.path.join(dst_dir, new_name)
                    shutil.copy(img_path, dst_path)


if __name__ == '__main__':
    # 数据集路径
    # 裁剪后的关键帧
    CASME2_onset_apex_offset_retinaface = '/kaggle/working/CASME2_onset_apex_offset_retinaface'
    # 光流图片
    CASME2_optflow_retinaface = '/kaggle/working/CASME2_optflow_retinaface'
    # 按情绪复制到对应文件夹
    data_folder_5 = '/kaggle/working/CASME2_retinaface_5'  # 5分类
    data_folder_3 = '/kaggle/working/CASME2_retinaface_3'  # 分类
    # 注释文件
    annotation_file = '/kaggle/input/casmeii/CASME2-coding-20140508.xlsx'
    CASME2_5c_3c(CASME2_onset_apex_offset_retinaface, CASME2_optflow_retinaface, data_folder_5, data_folder_3, annotation_file)
    zipPath = '/kaggle/working/CASME2_retinaface_5.zip'
    if os.path.exists(zipPath):
        os.remove(zipPath)
    zip_frames(data_folder_5, zipPath)
    print("打包完成")
    print(datetime.datetime.utcnow())
    print("CASMEⅡ 5分类 目录结构如下：\n")
    print_directory_structure(data_folder_5)
    zipPath = '/kaggle/working/CASME2_retinaface_3.zip'
    if os.path.exists(zipPath):
        os.remove(zipPath)
    zip_frames(data_folder_3, zipPath)
    print("打包完成")
    print(datetime.datetime.utcnow())
    print("CASMEⅡ 3分类 目录结构如下：\n")
    print_directory_structure(data_folder_3)
