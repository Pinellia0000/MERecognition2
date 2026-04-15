import os
import shutil
import pandas as pd
from tqdm import tqdm
import zipfile
import datetime
import glob


def delete_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"目录已删除: {path}")
    else:
        print(f"目录不存在: {path}")


def safe_parse_int(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def zip_frames(packagePath, zipPath):
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
    if is_root:
        print(f"{directory_name}目录结构如下：\n")
    items = sorted(os.listdir(root_dir))
    for idx, item in enumerate(items):
        path = os.path.join(root_dir, item)
        pointer = "└── " if idx == len(items) - 1 else "├── "
        print(indent + pointer + item)
        if os.path.isdir(path):
            extension = "    " if idx == len(items) - 1 else "│   "
            print_directory_structure(path, indent + extension, directory_name, is_root=False)


# ===================== 映射字典（全部正确） =====================
# 数字编号 → sxx 文件夹
subject_map = {
    "1": "s15",
    "2": "s16",
    "3": "s19",
    "4": "s20",
    "5": "s21",
    "6": "s22",
    "7": "s23",
    "8": "s24",
    "9": "s25",
    "10": "s26",
    "11": "s27",
    "12": "s29",
    "13": "s30",
    "14": "s31",
    "15": "s32",
    "16": "s33",
    "17": "s34",
    "18": "s35",
    "19": "s36",
    "20": "s37",
    "21": "s38",
    "22": "s40",
}

# 数字编号 → 纯数字（用于拼接 15_0102xxx）
subject_num_map = {
    "1": "15",
    "2": "16",
    "3": "19",
    "4": "20",
    "5": "21",
    "6": "22",
    "7": "23",
    "8": "24",
    "9": "25",
    "10": "26",
    "11": "27",
    "12": "29",
    "13": "30",
    "14": "31",
    "15": "32",
    "16": "33",
    "17": "34",
    "18": "35",
    "19": "36",
    "20": "37",
    "21": "38",
    "22": "40",
}

# 情绪前缀 → 编码
emotion_to_filename_code = {
    "disgust1": "0101",
    "disgust2": "0102",
    "anger1": "0401",
    "anger2": "0402",
    "happy1": "0502",
    "happy2": "0503",
    "happy3": "0505",
    "happy4": "0507",
    "happy5": "0508"
}

# 编码 → 文件夹后缀
filename_map = {
    "0101": "0101disgustingteeth",
    "0102": "0102eatingworms",
    "0401": "0401girlcrashing",
    "0402": "0402beatingpregnantwoman",
    "0502": "0502funnyerrors",
    "0503": "0503unnyfarting",
    "0505": "0505funnyinnovations",
    "0507": "0507climbingthewall",
    "0508": "0508funnydunkey"
}


def get_CASME_sq_onset_apex_offset(src_root, dst_root, excel_path):
    os.makedirs(dst_root, exist_ok=True)
    df = pd.read_excel(excel_path, header=None)  # 无表头

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing CASME_sq videos"):
        # ===================== 1. 读取Excel列 =====================
        label_str = str(row[0]).strip()  # 第0列：1,2...
        file_label = str(row[1]).strip()  # 第1列：disgust2_2
        onset = safe_parse_int(row[2])
        apex = safe_parse_int(row[3])
        offset = safe_parse_int(row[4])

        # 跳过无效帧
        if None in (onset, apex, offset):
            print(f"[SKIP] 跳过无效帧数据：{file_label}")
            continue

        # ===================== 2. 受试者映射 =====================
        subject = subject_map[label_str]  # s15
        subject_num = subject_num_map[label_str]  # 15

        # ===================== 3. 情绪文件夹映射 =====================
        # 示例：disgust2_2 → disgust2
        emotion_prefix = file_label.split('_')[0]
        code = emotion_to_filename_code[emotion_prefix]
        folder_suffix = filename_map[code]

        # 最终文件夹名：15_0102eatingworms
        final_folder = f"{subject_num}_{folder_suffix}"

        # ===================== 4. 路径拼接 =====================
        video_folder = os.path.join(src_root, subject, final_folder)
        dst_folder = os.path.join(dst_root, subject)
        os.makedirs(dst_folder, exist_ok=True)

        # ===================== 5. 复制关键帧 =====================
        for frame_type, frame_id in [('onset', onset), ('apex', apex), ('offset', offset)]:
            # ✅ 核心修复：小于3位自动补零 (1→001, 10→010, 100→100, 1000→1000)
            img_name = f"img{frame_id:03d}.jpg"

            src_img_path = os.path.join(video_folder, img_name)

            if os.path.exists(src_img_path):
                dst_img_name = f"{subject}_{file_label}_{frame_type}.jpg"
                dst_img_path = os.path.join(dst_folder, dst_img_name)
                shutil.copy(src_img_path, dst_img_path)
            else:
                print(f"[WARNING] 图片不存在：{src_img_path}")


if __name__ == "__main__":
    casme_sq_src_root = '/kaggle/input/datasets/garlic0000/casme-sq/rawpic/rawpic'
    casme2_sq_dst_root = '/kaggle/working/CASME_sq_onset_apex_offset'
    casme2_sq_excel_path = '/kaggle/input/datasets/garlic0000/casme-sq/CAS(ME)2code_final.xlsx'

    get_CASME_sq_onset_apex_offset(casme_sq_src_root, casme2_sq_dst_root, casme2_sq_excel_path)

    zipPath = '/kaggle/working/CASME_sq_onset_apex_offset.zip'
    zip_frames(casme2_sq_dst_root, zipPath)

    print_directory_structure(casme2_sq_dst_root, directory_name="CASME_sq_onset_apex_offset")