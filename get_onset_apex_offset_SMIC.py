import os
import shutil
import pandas as pd
from tqdm import tqdm  # 导入 tqdm 库
import zipfile
import datetime
import glob

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

def safe_parse_int(value):
    try:
        return int(value)
    except:
        return None

def get_SMIC_onset_apex_offset_auto(src_root, dst_root):
    """
    自动提取 SMIC 关键帧（不需要标注文件）
    规则：
    onset  = 第一帧
    apex   = 中间帧
    offset = 最后一帧
    输出结构和你原来的代码完全一致：subXX/xxx_onset/apex/offset.jpg
    """
    os.makedirs(dst_root, exist_ok=True)
    print(f"✅ 开始自动处理 SMIC 数据集：{src_root}")

    # 遍历 HS 下的所有受试者 s1, s2, s3...
    for s_folder in sorted(os.listdir(src_root)):
        s_path = os.path.join(src_root, s_folder)
        if not os.path.isdir(s_path):
            continue

        # 进入 micro 文件夹
        micro_path = os.path.join(s_path, "micro")
        if not os.path.exists(micro_path):
            continue

        # 遍历 3 个类别：negative, positive, surprise
        for emotion in os.listdir(micro_path):
            emo_path = os.path.join(micro_path, emotion)
            if not os.path.isdir(emo_path):
                continue

            # 遍历每个样本片段：s1_ne_01, s1_ne_02...
            for sample in sorted(os.listdir(emo_path)):
                sample_path = os.path.join(emo_path, sample)
                if not os.path.isdir(sample_path):
                    continue

                # 获取该片段所有帧
                frames = sorted([f for f in os.listdir(sample_path) if f.endswith(".bmp")])
                if len(frames) < 3:
                    print(f"[跳过] 帧数不足：{sample_path}")
                    continue

                # ====================== 关键帧自动计算 ======================
                onset_idx = 0                     # 第一帧
                apex_idx = len(frames) // 2        # 中间帧
                offset_idx = len(frames) - 1       # 最后一帧

                onset_img = frames[onset_idx]
                apex_img = frames[apex_idx]
                offset_img = frames[offset_idx]

                # 目标保存路径
                dst_sub_folder = os.path.join(dst_root, s_folder)
                os.makedirs(dst_sub_folder, exist_ok=True)

                # 复制 3 帧，命名规则和你原来代码完全一致
                for frame_type, img_name in [
                    ("onset", onset_img),
                    ("apex", apex_img),
                    ("offset", offset_img)
                ]:
                    src_img = os.path.join(sample_path, img_name)
                    dst_img_name = f"{sample}_{frame_type}.bmp"
                    dst_img = os.path.join(dst_sub_folder, dst_img_name)

                    shutil.copy(src_img, dst_img)

    print("✅ SMIC 关键帧提取完成！")

# ====================== 你只需要改这里 ======================
if __name__ == "__main__":
    # 你的 SMIC HS 路径
    SMIC_src_root = "/kaggle/input/datasets/garlic0000/smic-hs/HS"

    # 输出关键帧路径
    SMIC_dst_root = "/kaggle/working/SMIC_onset_apex_offset"

    # 开始提取
    get_SMIC_onset_apex_offset_auto(SMIC_src_root, SMIC_dst_root)
    # 打包
    zipPath = '/kaggle/working/SMIC_onset_apex_offset.zip'
    zip_frames(SMIC_dst_root, zipPath)
    # 输出关键帧结构
    print("CASME3关键帧目录结构如下：\n")
    print_directory_structure(SMIC_dst_root, directory_name="SMIC_onset_apex_offset")