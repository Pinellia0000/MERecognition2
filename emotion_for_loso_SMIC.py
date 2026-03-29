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


def SMIC_3c(SMIC_onset_apex_offset_retinaface, SMIC_optflow_retinaface, data_folder_3):
    """
    SMIC 数据集 无标注文件 → 根据文件名判断分类
    命名规则：
        po  → positive 积极 → 0
        ne  → negative 消极 → 1
        sur → surprise 惊讶 → 2
    同时处理：关键帧(onset/apex/offset) + 光流帧(u/v)
    """

    # 3分类标签
    label_dict_3 = {
        'po': 0,
        'ne': 1,
        'sur': 2
    }

    # 创建输出目录 0,1,2
    for label in sorted(set(label_dict_3.values())):
        os.makedirs(os.path.join(data_folder_3, str(label)), exist_ok=True)

    total_count = {'0': 0, '1': 0, '2': 0}

    # -------------------------- 统一处理函数：遍历所有图片，按文件名分类 --------------------------
    def process_folder(src_root):
        """遍历目录下所有图片，根据文件名前缀分配到对应标签文件夹"""
        if not os.path.exists(src_root):
            print(f"⚠️  路径不存在: {src_root}")
            return

        # 递归遍历所有子目录
        for root, dirs, files in os.walk(src_root):
            for img_name in tqdm(files, desc=f"处理 {os.path.basename(src_root)}"):
                # 只处理图片
                if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                    continue

                # ========== 核心：根据文件名前缀判断情绪 ==========
                img_lower = img_name.lower()
                label_id = None

                if 'po' in img_lower:
                    label_id = 0
                elif 'ne' in img_lower:
                    label_id = 1
                elif 'sur' in img_lower:
                    label_id = 2
                else:
                    # 不匹配的文件跳过
                    continue

                # 复制到对应标签目录
                target_dir = os.path.join(data_folder_3, str(label_id))
                src_path = os.path.join(root, img_name)
                dst_path = os.path.join(target_dir, img_name)

                if not os.path.exists(dst_path):
                    shutil.copy(src_path, dst_path)
                    total_count[str(label_id)] += 1

    # 处理 关键帧
    print("\n📌 开始处理 SMIC 关键帧...")
    process_folder(SMIC_onset_apex_offset_retinaface)

    # 处理 光流帧
    print("\n📌 开始处理 SMIC 光流帧...")
    process_folder(SMIC_optflow_retinaface)

    # 输出统计
    print("\n🎉 SMIC 3分类整理完成！")
    print(f"✅ 积极 (0)：{total_count['0']} 张")
    print(f"✅ 消极 (1)：{total_count['1']} 张")
    print(f"✅ 惊讶 (2)：{total_count['2']} 张")
    print(f"📊 总计：{sum(total_count.values())} 张")


if __name__ == '__main__':
    # ===================== SMIC 数据集（按文件名 po/ne/sur 分类）=====================
    SMIC_onset_apex_offset_retinaface = "/kaggle/working/SMIC_onset_apex_offset_retinaface"
    SMIC_optflow_retinaface = "/kaggle/working/SMIC_optflow_retinaface"
    SMIC_data_folder_3 = "/kaggle/working/SMIC_retinaface_3"

    # 执行分类
    SMIC_3c(SMIC_onset_apex_offset_retinaface, SMIC_optflow_retinaface, SMIC_data_folder_3)

    # 打包
    zipPath = '/kaggle/working/SMIC_retinaface_3.zip'
    zip_frames(SMIC_data_folder_3, zipPath)

    # 打印目录结构
    print_directory_structure(SMIC_data_folder_3, directory_name='SMIC_retinaface_3')