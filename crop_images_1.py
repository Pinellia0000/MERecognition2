import glob
import cv2
from RetinaFace.tools import FaceDetector
import os
from tqdm import tqdm  # 添加进度条支持
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


def crop_images_retinaface(src_root_path, dst_root_path):
    """
    获取关键帧之后 结构类似 可以通用
    """
    # 模型路径和初始化
    face_det_model_path = "/kaggle/input/retinaface-model/retinaface_Resnet50_Final.pth"
    face_detection = FaceDetector(face_det_model_path)

    # 遍历 subXX 文件夹
    subject_folders = [f for f in os.listdir(src_root_path) if os.path.isdir(os.path.join(src_root_path, f))]

    for sub_folder_name in tqdm(subject_folders, desc="Processing subjects"):
        sub_folder_path = os.path.join(src_root_path, sub_folder_name)

        # 直接遍历 subXX 下的图片（两层结构）
        image_paths = sorted(glob.glob(os.path.join(sub_folder_path, '*.jpg')))
        index = 0
        face_left = face_top = face_right = face_bottom = 0

        for img_file_path in tqdm(image_paths, desc=f"  {sub_folder_name}", leave=False):
            image = cv2.imread(img_file_path)
            if image is None:
                print(f"[WARNING] Cannot read image: {img_file_path}")
                continue

            # 第一个图做人脸检测
            if index == 0:
                face_left, face_top, face_right, face_bottom = face_detection.cal(image)

            # 裁剪人脸
            face = image[face_top:face_bottom + 1, face_left:face_right + 1, :]
            face = cv2.resize(face, (128, 128))

            # 构造保存路径，保持原有结构
            relative_path = os.path.relpath(img_file_path, src_root_path)
            dst_img_path = os.path.join(dst_root_path, relative_path)
            dst_folder = os.path.dirname(dst_img_path)
            os.makedirs(dst_folder, exist_ok=True)

            cv2.imwrite(dst_img_path, face)
            index += 1

    print("Face cropping and saving complete.")




if __name__ == '__main__':
    # CASMEⅡ 数据集
    # 原始图片路径（已保留结构：subXX/EPXX_xxf/）
    casme2_src_root_path = "/kaggle/working/CASME2_onset_apex_offset"
    # 新保存路径
    casme2_dst_root_path = "/kaggle/working/CASME2_onset_apex_offset_retinaface"
    crop_images_retinaface(casme2_src_root_path, casme2_dst_root_path)
    zipPath = '/kaggle/working/CASME2_onset_apex_offset_retinaface.zip'
    if os.path.exists(zipPath):
        os.remove(zipPath)
    zip_frames(casme2_dst_root_path, zipPath)
    print("打包完成")
    print(datetime.datetime.utcnow())
    print("目录结构如下：\n")
    print_directory_structure(casme2_dst_root_path)

    # SAMM 数据集
    # 原始图片路径（已保留结构：）
    samm_src_root_path = "/kaggle/working/SAMM_onset_apex_offset"
    # 新保存路径
    samm_dst_root_path = "/kaggle/working/SAMM_onset_apex_offset_retinaface"
    crop_images_retinaface(samm_src_root_path, samm_dst_root_path)
    zipPath = '/kaggle/working/SAMM_onset_apex_offset_retinaface.zip'
    if os.path.exists(zipPath):
        os.remove(zipPath)
    zip_frames(samm_dst_root_path, zipPath)
    print("打包完成")
    print(datetime.datetime.utcnow())
    print("目录结构如下：\n")
    print_directory_structure(samm_dst_root_path)

    # CAS(ME)^3 数据集
    # 原始图片路径（已保留结构：）
    casme3_src_root_path = "/kaggle/working/CASME3_onset_apex_offset"
    # 新保存路径
    casme3_dst_root_path = "/kaggle/working/CASME3_onset_apex_offset_retinaface"
    crop_images_retinaface(casme3_src_root_path, casme3_dst_root_path)
    zipPath = '/kaggle/working/CASME3_onset_apex_offset_retinaface.zip'
    if os.path.exists(zipPath):
        os.remove(zipPath)
    zip_frames(casme3_dst_root_path, zipPath)
    print("打包完成")
    print(datetime.datetime.utcnow())
    print("目录结构如下：\n")
    print_directory_structure(casme3_dst_root_path)
