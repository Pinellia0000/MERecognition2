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


def CASME2_5c_3c(CASME2_onset_apex_offset_retinaface, CASME2_optflow_retinaface,
                 data_folder_5, data_folder_3, annotation_file):
    """
    关键帧结构
    ├── sub01
    │   ├── sub01_EP02_01f_apex.jpg
    │   ├── sub01_EP02_01f_offset.jpg
    │   ├── sub01_EP02_01f_onset.jpg
    │   ├── sub01_EP03_02_apex.jpg
    光流帧结构
    ├── sub01
    │   ├── sub01_EP02_01f_1_u.jpg
    │   ├── sub01_EP02_01f_1_v.jpg
    │   ├── sub01_EP02_01f_2_u.jpg
    │   ├── sub01_EP02_01f_2_v.jpg

    CASME2 5分类 + 3分类 整理
    sadness fear 数量太少 不参与分类
    5类：'happiness':0, 'surprise':1, 'disgust':2, 'repression':3, 'others':4
    3类：'positive':0, 'negative':1, 'surprise':2
         positive = happiness
         negative = disgust + repression
         surprise = surprise
         others 不参与 3分类
    """

    # 5分类字典
    label_dict_5 = {
        'happiness': 0,
        'surprise': 1,
        'disgust': 2,
        'repression': 3,
        'others': 4
    }

    # 3分类字典（没有 others）
    label_dict_3 = {
        'happiness': 0,  # positive
        'disgust': 1,  # negative
        'repression': 1,  # negative
        'surprise': 2  # surprise
    }

    # 创建 5分类目录
    for label in sorted(set(label_dict_5.values())):
        os.makedirs(os.path.join(data_folder_5, str(label)), exist_ok=True)

    # 创建 3分类目录（不包含 others）
    for label in sorted(set(label_dict_3.values())):
        os.makedirs(os.path.join(data_folder_3, str(label)), exist_ok=True)

    # 读取注释文件
    anno_df = pd.read_excel(annotation_file)

    # 遍历所有被试目录
    for subject in tqdm(os.listdir(CASME2_onset_apex_offset_retinaface), desc="Processing CASME2 subjects"):
        sub_folder_path = os.path.join(CASME2_onset_apex_offset_retinaface, subject)
        if not os.path.isdir(sub_folder_path):
            continue

        # 筛选注释行 (Subject 列是 '01','02'...)
        sub_df = anno_df[anno_df['Subject'].astype(str).str.zfill(2) == subject.replace("sub", "")]
        if sub_df.empty:
            print(f"[WARNING] {subject} 在注释文件中没有匹配到任何行")
            continue

        def process_and_copy(src_folder):
            if not os.path.exists(src_folder):
                return
            for img_name in os.listdir(src_folder):
                img_path = os.path.join(src_folder, img_name)

                # Filename 必须包含在图片名里
                matched_rows = sub_df[sub_df['Filename'].astype(str).apply(lambda x: x in img_name)]
                if matched_rows.empty:
                    continue

                for _, row in matched_rows.iterrows():
                    emotion = str(row['Estimated Emotion']).strip().lower()
                    if emotion not in label_dict_5:
                        continue

                    # 5分类
                    dst_path_5 = os.path.join(data_folder_5, str(label_dict_5[emotion]), img_name)
                    if not os.path.exists(dst_path_5):
                        shutil.copy(img_path, dst_path_5)

                    # 3分类 (others 不参与)
                    if emotion in label_dict_3:
                        dst_path_3 = os.path.join(data_folder_3, str(label_dict_3[emotion]), img_name)
                        if not os.path.exists(dst_path_3):
                            shutil.copy(img_path, dst_path_3)

        # 关键帧和光流都处理
        process_and_copy(sub_folder_path)
        process_and_copy(os.path.join(CASME2_optflow_retinaface, subject))


def SAMM_3c(SAMM_onset_apex_offset_retinaface, SAMM_optflow_retinaface,
            data_folder_3, annotation_file):
    """
    关键帧结构
    ├── 006
    │   ├── 006_1_2_apex.jpg
    │   ├── 006_1_2_offset.jpg
    │   ├── 006_1_2_onset.jpg
    │   ├── 006_1_3_apex.jpg
    光流帧结构
    ├── 006
    │   ├── 006_1_2_1_u.jpg
    │   ├── 006_1_2_1_v.jpg
    │   ├── 006_1_2_2_u.jpg
    │   ├── 006_1_2_2_v.jpg

    Other 不参与 3分类

    SAMM 3分类 整理
    3类：'positive':0, 'negative':1, 'surprise':2
         positive = Happiness
         negative = Anger + Disgust + Contempt + Sadness + Fear
         surprise = Surprise
    """

    # 3分类字典（不包含 'Other'）
    label_dict_3 = {
        'Happiness': 0,
        'Anger': 1, 'Disgust': 1, 'Contempt': 1, 'Sadness': 1, 'Fear': 1,
        'Surprise': 2
    }

    # 创建输出目录
    for label in sorted(set(label_dict_3.values())):
        os.makedirs(os.path.join(data_folder_3, str(label)), exist_ok=True)

    # 注意：samm 数据不是从第一行开始，前几行有说明性文字
    anno_df = pd.read_excel(annotation_file, header=13)  # 列名在第14行

    # 遍历被试文件夹
    for subject in tqdm(os.listdir(SAMM_onset_apex_offset_retinaface), desc="Processing SAMM"):
        sub_folder_path = os.path.join(SAMM_onset_apex_offset_retinaface, subject)
        if not os.path.isdir(sub_folder_path):
            continue

        # 找出该被试的注释行
        sub_df = anno_df[anno_df['Subject'] == int(subject)]
        if sub_df.empty:
            print(f"[WARNING] {subject} 在注释文件中未找到")
            continue

        def process_and_copy(src_folder):
            """处理关键帧/光流帧，按注释匹配并复制到对应类别"""
            if not os.path.exists(src_folder):
                return
            for img_name in os.listdir(src_folder):
                img_path = os.path.join(src_folder, img_name)

                # 图片名称中如果包含注释文件中的 Filename
                matched_rows = sub_df[sub_df['Filename'].astype(str).apply(lambda x: x in img_name)]
                if matched_rows.empty:
                    continue

                for _, row in matched_rows.iterrows():
                    # 获取该行的 Estimated Emotion
                    emotion = str(row['Estimated Emotion']).strip()

                    # Other 不参与分类
                    if emotion not in label_dict_3:
                        # Debug 打印
                        # print(f"[SKIP] {subject}_{img_name} -> {emotion} (不参与3分类)")
                        continue

                    # 复制文件
                    label_id = label_dict_3[emotion]
                    dst_dir = os.path.join(data_folder_3, str(label_id))
                    dst_path = os.path.join(dst_dir, img_name)
                    if not os.path.exists(dst_path):  # 避免重复复制
                        shutil.copy(img_path, dst_path)

        # 处理关键帧
        process_and_copy(sub_folder_path)
        # 处理光流帧
        process_and_copy(os.path.join(SAMM_optflow_retinaface, subject))


def CASME3_7c_4c_3c(CASME3_onset_apex_offset_retinaface, CASME3_optflow_retinaface,
                    data_folder_7, data_folder_4, data_folder_3, annotation_file):
    """
    关键帧结构
    ├── spNO.1
    │   ├── spNO.1_a_355_apex.jpg
    │   ├── spNO.1_a_355_offset.jpg
    │   ├── spNO.1_a_355_onset.jpg
    │   ├── spNO.1_b_166_apex.jpg
    │   ├── spNO.1_b_166_offset.jpg
    光流帧结构
    ├── spNO.1
    │   ├── spNO.1_a_355_1_u.jpg
    │   ├── spNO.1_a_355_1_v.jpg
    │   ├── spNO.1_a_355_2_u.jpg
    │   ├── spNO.1_a_355_2_v.jpg

    CASME3 分类整理
    7类：'happy':0, 'surprise':1, 'disgust':2, 'anger':3, 'fear':4, 'sad':5, 'others':6

    4类（others 参与）：
        'positive':0, 'negative':1, 'surprise':2, 'others':3
        positive = happy
        negative = disgust + anger + fear + sad
        surprise = surprise
        others = others

    3类（others 不参与）：
        'positive':0, 'negative':1, 'surprise':2
        positive = happy
        negative = disgust + anger + fear + sad
        surprise = surprise
    """

    # 标签字典
    label_dict_7 = {
        'happy': 0, 'surprise': 1, 'disgust': 2,
        'anger': 3, 'fear': 4, 'sad': 5, 'others': 6
    }
    label_dict_4 = {
        'happy': 0, 'disgust': 1, 'anger': 1, 'fear': 1, 'sad': 1,
        'surprise': 2, 'others': 3
    }
    label_dict_3 = {
        'happy': 0, 'disgust': 1, 'anger': 1, 'fear': 1, 'sad': 1,
        'surprise': 2
    }

    # 创建输出目录
    for label in sorted(set(label_dict_7.values())):
        os.makedirs(os.path.join(data_folder_7, str(label)), exist_ok=True)
    for label in sorted(set(label_dict_4.values())):
        os.makedirs(os.path.join(data_folder_4, str(label)), exist_ok=True)
    for label in sorted(set(label_dict_3.values())):
        os.makedirs(os.path.join(data_folder_3, str(label)), exist_ok=True)

    # 读取注释文件
    anno_df = pd.read_excel(annotation_file)

    # 遍历被试
    for subject in tqdm(os.listdir(CASME3_onset_apex_offset_retinaface), desc="Processing CASME3"):
        sub_folder_path = os.path.join(CASME3_onset_apex_offset_retinaface, subject)
        if not os.path.isdir(sub_folder_path):
            continue

        # 找出该被试的注释行
        sub_df = anno_df[anno_df['Subject'] == subject]
        if sub_df.empty:
            print(f"[WARNING] {subject} 在注释文件中未找到")
            continue

        def process_and_copy(src_folder):
            """处理关键帧/光流帧，按注释匹配并复制到对应类别"""
            if not os.path.exists(src_folder):
                return
            for img_name in os.listdir(src_folder):
                img_path = os.path.join(src_folder, img_name)

                # 图片名称中如果包含注释文件中的 Filename
                matched_rows = sub_df[sub_df['Filename'].astype(str).apply(lambda x: x in img_name)]
                if matched_rows.empty:
                    continue

                for _, row in matched_rows.iterrows():
                    # 获取 emotion
                    emotion = str(row['emotion']).strip().lower()
                    if emotion not in label_dict_7:
                        # print(f"[SKIP] {subject}_{img_name} -> {emotion} (不参与分类)")
                        continue

                    # ---- 7类 ----
                    dst_path_7 = os.path.join(data_folder_7, str(label_dict_7[emotion]), img_name)
                    if not os.path.exists(dst_path_7):
                        shutil.copy(img_path, dst_path_7)

                    # ---- 4类 ----
                    if emotion in label_dict_4:
                        dst_path_4 = os.path.join(data_folder_4, str(label_dict_4[emotion]), img_name)
                        if not os.path.exists(dst_path_4):
                            shutil.copy(img_path, dst_path_4)

                    # ---- 3类 ---- (others 不参与)
                    if emotion in label_dict_3:
                        dst_path_3 = os.path.join(data_folder_3, str(label_dict_3[emotion]), img_name)
                        if not os.path.exists(dst_path_3):
                            shutil.copy(img_path, dst_path_3)

        # 处理关键帧
        process_and_copy(sub_folder_path)
        # 处理光流帧
        process_and_copy(os.path.join(CASME3_optflow_retinaface, subject))


def delete_main_1():
    casme2_dst_root = '/kaggle/working/CASME2_onset_apex_offset'
    samm_dst_root = '/kaggle/working/SAMM_onset_apex_offset'
    casme3_dst_root = '/kaggle/working/CASME3_onset_apex_offset'
    delete_directory(casme2_dst_root)
    delete_directory(samm_dst_root)
    delete_directory(casme3_dst_root)


if __name__ == '__main__':
    # # 减少一些目录
    # delete_main_1()
    # CASMEⅡ
    # 数据集路径
    CASME2_onset_apex_offset_retinaface = '/kaggle/working/CASME2_onset_apex_offset_retinaface'
    CASME2_optflow_retinaface = '/kaggle/working/CASME2_optflow_retinaface'
    data_folder_5 = '/kaggle/working/CASME2_retinaface_5'
    data_folder_3 = '/kaggle/working/CASME2_retinaface_3'
    annotation_file = '/kaggle/input/datasets/garlic0000/casmeii/CASME2-coding-20140508.xlsx'

    # 整理数据
    CASME2_5c_3c(CASME2_onset_apex_offset_retinaface, CASME2_optflow_retinaface,
                 data_folder_5, data_folder_3, annotation_file)

    # 打包 5分类
    zipPath = '/kaggle/working/CASME2_retinaface_5.zip'
    zip_frames(data_folder_5, zipPath)
    print_directory_structure(data_folder_5, directory_name='CASME2_retinaface_5')

    # 打包 3分类
    zipPath = '/kaggle/working/CASME2_retinaface_3.zip'
    zip_frames(data_folder_3, zipPath)
    print_directory_structure(data_folder_3, directory_name='CASME2_retinaface_3')

    # # SAMM
    # # 数据集路径
    # SAMM_onset_apex_offset_retinaface = '/kaggle/working/SAMM_onset_apex_offset_retinaface'
    # SAMM_optflow_retinaface = '/kaggle/working/SAMM_optflow_retinaface'
    # data_folder_3 = '/kaggle/working/SAMM_retinaface_3'
    # annotation_file = '/kaggle/input/samm-dataset/SAMM/SAMM_Micro_FACS_Codes_v2.xlsx'
    #
    # # 整理数据
    # SAMM_3c(SAMM_onset_apex_offset_retinaface, SAMM_optflow_retinaface,
    #         data_folder_3, annotation_file)
    #
    # # 打包 3分类
    # zipPath = '/kaggle/working/SAMM_retinaface_3.zip'
    # zip_frames(data_folder_3, zipPath)
    # print_directory_structure(data_folder_3, directory_name='SAMM_retinaface_3')
    #
    # # CASME3
    # # 数据集路径
    # CASME3_onset_apex_offset_retinaface = '/kaggle/working/CASME3_onset_apex_offset_retinaface'
    # CASME3_optflow_retinaface = '/kaggle/working/CASME3_optflow_retinaface'
    # data_folder_7 = '/kaggle/working/CASME3_retinaface_7'
    # data_folder_4 = '/kaggle/working/CASME3_retinaface_4'
    # data_folder_3 = '/kaggle/working/CASME3_retinaface_3'
    # annotation_file = '/kaggle/input/casme3/cas(me)3_part_A_ME_label_JpgIndex_v2_20250903.xlsx'
    #
    # # 整理数据
    # CASME3_7c_4c_3c(CASME3_onset_apex_offset_retinaface, CASME3_optflow_retinaface,
    #                 data_folder_7, data_folder_4, data_folder_3, annotation_file)
    # # 打包 7分类
    # zipPath = '/kaggle/working/CASME2_retinaface_7.zip'
    # zip_frames(data_folder_7, zipPath)
    # print_directory_structure(data_folder_7, directory_name='CASME2_retinaface_7')
    #
    # # 打包 4分类
    # zipPath = '/kaggle/working/CASME3_retinaface_4.zip'
    # zip_frames(data_folder_4, zipPath)
    # print_directory_structure(data_folder_4, directory_name='CASME3_retinaface_4')
    #
    # # 打包 3分类
    # zipPath = '/kaggle/working/CASME3_retinaface_3.zip'
    # zip_frames(data_folder_3, zipPath)
    # print_directory_structure(data_folder_3, directory_name='CASME3_retinaface_3')
