import glob
import cv2
from RetinaFace.tools import FaceDetector
import os
from tqdm import tqdm
import zipfile
import datetime
import shutil


def delete_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"目录已删除: {path}")
    else:
        print(f"目录不存在: {path}")


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


def crop_images_retinaface(src_root_path, dst_root_path):
    face_det_model_path = "/kaggle/input/retinaface-model/retinaface_Resnet50_Final.pth"
    face_detection = FaceDetector(face_det_model_path)

    subject_folders = [f for f in os.listdir(src_root_path) if os.path.isdir(os.path.join(src_root_path, f))]

    for sub_folder_name in tqdm(subject_folders, desc="Processing subjects"):
        sub_folder_path = os.path.join(src_root_path, sub_folder_name)
        image_paths = sorted(glob.glob(os.path.join(sub_folder_path, '*.jpg')))
        index = 0
        face_left = face_top = face_right = face_bottom = 0

        for img_file_path in tqdm(image_paths, desc=f"  {sub_folder_name}", leave=False):
            image = cv2.imread(img_file_path)
            if image is None:
                print(f"[WARNING] Cannot read image: {img_file_path}")
                continue

            if index == 0:
                face_left, face_top, face_right, face_bottom = face_detection.cal(image)

            face = image[face_top:face_bottom + 1, face_left:face_right + 1, :]

            if face is None or face.size == 0:
                face = image
                print(f"[原图] 使用原图: {img_file_path}")

            try:
                face = cv2.resize(face, (128, 128))
            except Exception as e:
                print(f"[ERROR] Resize failed: {e}")
                continue

            relative_path = os.path.relpath(img_file_path, src_root_path)
            dst_img_path = os.path.join(dst_root_path, relative_path)
            dst_folder = os.path.dirname(dst_img_path)
            os.makedirs(dst_folder, exist_ok=True)

            cv2.imwrite(dst_img_path, face)
            index += 1

    print("Face cropping complete.")


def resize_only_images(src_root_path, dst_root_path):
    """
    SMIC专用：仅缩放 128x128，不做人脸检测
    """
    subject_folders = [f for f in os.listdir(src_root_path) if os.path.isdir(os.path.join(src_root_path, f))]

    for sub_folder_name in tqdm(subject_folders, desc="Processing SMIC subjects"):
        sub_folder_path = os.path.join(src_root_path, sub_folder_name)
        image_paths = sorted(glob.glob(os.path.join(sub_folder_path, '*.jpg')))

        for img_file_path in tqdm(image_paths, desc=f"  {sub_folder_name}", leave=False):
            image = cv2.imread(img_file_path)
            if image is None:
                print(f"[WARNING] Cannot read: {img_file_path}")
                continue

            try:
                resized_img = cv2.resize(image, (128, 128))
            except Exception as e:
                print(f"[ERROR] Resize failed: {e}")
                continue

            relative_path = os.path.relpath(img_file_path, src_root_path)
            dst_img_path = os.path.join(dst_root_path, relative_path)
            dst_folder = os.path.dirname(dst_img_path)
            os.makedirs(dst_folder, exist_ok=True)

            cv2.imwrite(dst_img_path, resized_img)

    print("SMIC resize 128x128 done.")


if __name__ == '__main__':

    # ========== SMIC（仅缩放 128x128）==========
    smic_src = "/kaggle/working/SMIC_onset_apex_offset"
    smic_dst = "/kaggle/working/SMIC_onset_apex_offset_retinaface"
    resize_only_images(smic_src, smic_dst)
    zip_frames(smic_dst, '/kaggle/working/SMIC_onset_apex_offset_retinaface.zip')
    print_directory_structure(smic_dst, directory_name='SMIC')