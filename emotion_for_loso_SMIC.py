import os
import shutil
import pandas as pd
from tqdm import tqdm
import zipfile
import datetime


def delete_directory(path):
    """
    删除指定目录及以下所有文件
    path: 要删除的目录路径
    """
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"目录已删除: {path}")
    else:
        print(f"目录不存在: {path}")


def zip_frames(packagePath, zipPath):
    """
    packagePath: 文件夹路径
    zipPath: 压缩包路径
    """
    if os.path.exists(zipPath):
        os.remove(zipPath)
    zip = zipfile.ZipFile(zipPath, 'w', zipfile.ZIP_DEFLATED)
    for path, dirNames, fileNames in os.walk(packagePath):
        fpath = path.replace(packagePath, '')
        for name in fileNames:
            fullName = os.path.join(path, name)
            name = os.path.join(fpath, name)
            zip.write(fullName, name)
    zip.close()
    print("打包完成")
    print(datetime.datetime.utcnow())


def print_directory_structure(root_dir, indent="", directory_name="", is_root=True):
    """
    递归打印目录结构
    """
    if is_root:
        print(f"{directory_name}目录结构如下：\n")
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
            print_directory_structure(path, indent + extension, directory_name, is_root=False)


def SMIC_3c(SMIC_onset_apex_offset_resize128, SMIC_optflow_retinaface, data_folder_3):
    """
    SMIC 数据集 没有标注文件，直接按文件夹名分类
    文件夹名就是标签：
        positive → 0
        negative → 1
        surprise → 2
    """

    # SMIC 3分类标签
    label_dict_3 = {
        'positive': 0,
        'negative': 1,
        'surprise': 2
    }

    # 创建输出目录
    for label in sorted(set(label_dict_3.values())):
        os.makedirs(os.path.join(data_folder_3, str(label)), exist_ok=True)

    # 遍历处理每个类别
    for label_name in ['positive', 'negative', 'surprise']:
        label_id = label_dict_3[label_name]
        target_dir = os.path.join(data_folder_3, str(label_id))
        count = 0

        print(f"\n→ 处理 SMIC 类别：{label_name} -> {label_id}")

        # -------------------------- 关键修复：遍历所有子目录复制图片 --------------------------
        # 处理关键帧
        key_frame_folder = os.path.join(SMIC_onset_apex_offset_resize128, label_name)
        if os.path.exists(key_frame_folder):
            print(f"  关键帧路径：{key_frame_folder}")
            for root, dirs, files in os.walk(key_frame_folder):
                for img_name in tqdm(files, desc=f"{label_name} 关键帧"):
                    if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                        src = os.path.join(root, img_name)
                        dst = os.path.join(target_dir, img_name)
                        shutil.copy(src, dst)
                        count += 1
        else:
            print(f"  ⚠️  关键帧路径不存在：{key_frame_folder}")

        # 处理光流帧
        optflow_folder = os.path.join(SMIC_optflow_retinaface, label_name)
        if os.path.exists(optflow_folder):
            print(f"  光流路径：{optflow_folder}")
            for root, dirs, files in os.walk(optflow_folder):
                for img_name in tqdm(files, desc=f"{label_name} 光流帧"):
                    if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                        src = os.path.join(root, img_name)
                        dst = os.path.join(target_dir, img_name)
                        shutil.copy(src, dst)
                        count += 1
        else:
            print(f"  ⚠️  光流路径不存在：{optflow_folder}")

        print(f"  ✅ 该类别总计复制图片：{count} 张")

    print("\n🎉 SMIC 3分类整理完成！")


def delete_main_1():
    casme2_dst_root = '/kaggle/working/CASME2_onset_apex_offset'
    samm_dst_root = '/kaggle/working/SAMM_onset_apex_offset'
    casme3_dst_root = '/kaggle/working/CASME3_onset_apex_offset'
    delete_directory(casme2_dst_root)
    delete_directory(samm_dst_root)
    delete_directory(casme3_dst_root)


if __name__ == '__main__':
    # ===================== SMIC 数据集（无标注，按文件夹分类）=====================
    SMIC_onset_apex_offset_retinaface = "/kaggle/working/SMIC_onset_apex_offset_retinaface"
    SMIC_optflow_retinaface = "/kaggle/working/SMIC_optflow_retinaface"
    SMIC_data_folder_3 = "/kaggle/working/SMIC_retinaface_3"

    # 分类
    SMIC_3c(SMIC_onset_apex_offset_retinaface, SMIC_optflow_retinaface, SMIC_data_folder_3)

    # 打包
    zipPath = '/kaggle/working/SMIC_retinaface_3.zip'
    zip_frames(SMIC_data_folder_3, zipPath)

    # 打印目录结构
    print_directory_structure(SMIC_data_folder_3, directory_name='SMIC_retinaface_3')