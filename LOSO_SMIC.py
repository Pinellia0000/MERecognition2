import os
import shutil
import zipfile
import datetime
from tqdm import tqdm
from collections import defaultdict


def zip_subject(subject_folder, temp_zip_folder):
    os.makedirs(temp_zip_folder, exist_ok=True)
    subject_name = os.path.basename(subject_folder)
    zip_path = os.path.join(temp_zip_folder, f"{subject_name}.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as zipf:
        for root, dirs, files in os.walk(subject_folder):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, subject_folder)
                zipf.write(file_path, arcname)
    shutil.rmtree(subject_folder)
    return zip_path


def merge_subject_zips(temp_zip_folder, final_zip):
    zip_files = sorted(os.listdir(temp_zip_folder))
    with zipfile.ZipFile(final_zip, 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as final_zipf:
        for zf_name in zip_files:
            zf_path = os.path.join(temp_zip_folder, zf_name)
            with zipfile.ZipFile(zf_path, 'r') as sub_zip:
                for file in sub_zip.namelist():
                    data = sub_zip.read(file)
                    arcname = os.path.join(zf_name.replace(".zip", ""), file).replace("\\", "/")
                    final_zipf.writestr(arcname, data)
            os.remove(zf_path)


def copy_file_no_tmp(src, dst, buffer_size=1024 * 1024):
    try:
        with open(src, 'rb') as fsrc, open(dst, 'wb') as fdst:
            while True:
                buf = fsrc.read(buffer_size)
                if not buf:
                    break
                fdst.write(buf)
    except OSError as e:
        if e.errno == 28:
            print(f"[ERROR] 磁盘空间不足，拷贝失败: {src}")
            print_disk_usage("/kaggle/working")
            raise
        else:
            raise


def print_disk_usage(path="/kaggle/working"):
    usage = shutil.disk_usage(path)
    total_gb = usage.total / (1024 ** 3)
    used_gb = usage.used / (1024 ** 3)
    free_gb = usage.free / (1024 ** 3)
    print(f"磁盘路径: {path}")
    print(f"总容量: {total_gb:.2f} GB")
    print(f"已用: {used_gb:.2f} GB")
    print(f"可用: {free_gb:.2f} GB")


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


def zip_frames_stream(packagePath, zipPath):
    if os.path.exists(zipPath):
        os.remove(zipPath)
    with zipfile.ZipFile(zipPath, 'w', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zipf:
        for root, dirs, files in os.walk(packagePath):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, packagePath).replace("\\", "/")
                zipf.write(file_path, arcname)
    print("打包完成")
    print(datetime.datetime.utcnow())


def print_zip_structure(zip_path):
    print(f"{zip_path} 压缩包内容结构如下：\n")
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        all_files = sorted(zipf.namelist())
        tree = lambda: defaultdict(tree)
        root = tree()
        for f in all_files:
            parts = f.strip("/").split("/")
            node = root
            for part in parts:
                node = node[part]

        def _print(node, indent="", is_root=True):
            items = sorted(node.keys())
            for idx, item in enumerate(items):
                pointer = "└── " if idx == len(items) - 1 else "├── "
                print(indent + pointer + item)
                if node[item]:
                    extension = "    " if idx == len(items) - 1 else "│   "
                    _print(node[item], indent + extension, is_root=False)

        _print(root)


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


def process_loso_each(data_folder, loso_folder, num_classes, dataset_name="Dataset"):
    subjects = set()
    for class_folder in range(num_classes):
        class_path = os.path.join(data_folder, str(class_folder))
        if not os.path.exists(class_path):
            continue
        for file in os.listdir(class_path):
            if file.startswith("sub") or file.startswith("spNO"):
                subjects.add(file.split("_")[0])
            else:
                subjects.add(file.split("_")[0].zfill(3))
    subjects = sorted(subjects)
    for subject in tqdm(subjects, desc=f"Processing {dataset_name} subjects"):
        sub_folder = os.path.join(loso_folder, subject)
        os.makedirs(sub_folder, exist_ok=True)
        for class_folder in range(num_classes):
            class_path = os.path.join(data_folder, str(class_folder))
            if not os.path.exists(class_path):
                continue
            files = [file for file in os.listdir(class_path) if file.startswith(subject)]
            not_files = [file for file in os.listdir(class_path) if not file.startswith(subject)]
            if files:
                test_folder = os.path.join(sub_folder, 'test', str(class_folder))
                os.makedirs(test_folder, exist_ok=True)
                for file in files:
                    shutil.copy(os.path.join(class_path, file), os.path.join(test_folder, file))
            train_folder = os.path.join(sub_folder, 'train', str(class_folder))
            os.makedirs(train_folder, exist_ok=True)
            for file in not_files:
                shutil.copy(os.path.join(class_path, file), os.path.join(train_folder, file))


def process_loso_each_CASME3(data_folder, loso_folder, num_classes, dataset_name="Dataset"):
    subjects = set()
    for class_folder in range(num_classes):
        class_path = os.path.join(data_folder, str(class_folder))
        if not os.path.exists(class_path):
            continue
        for file in os.listdir(class_path):
            if file.startswith("sub") or file.startswith("spNO"):
                subjects.add(file.split("_")[0])
            else:
                subjects.add(file.split("_")[0].zfill(3))
    subjects = sorted(subjects)
    temp_zip_folder = os.path.join(loso_folder, "_temp_subject_zips")
    os.makedirs(temp_zip_folder, exist_ok=True)
    for subject in tqdm(subjects, desc=f"Processing {dataset_name} subjects"):
        sub_folder = os.path.join(loso_folder, subject)
        os.makedirs(sub_folder, exist_ok=True)
        for class_folder in range(num_classes):
            class_path = os.path.join(data_folder, str(class_folder))
            if not os.path.exists(class_path):
                continue
            files = [file for file in os.listdir(class_path) if file.startswith(subject)]
            not_files = [file for file in os.listdir(class_path) if not file.startswith(subject)]
            if files:
                test_folder = os.path.join(sub_folder, 'test', str(class_folder))
                os.makedirs(test_folder, exist_ok=True)
                for file in files:
                    copy_file_no_tmp(os.path.join(class_path, file), os.path.join(test_folder, file))
            train_folder = os.path.join(sub_folder, 'train', str(class_folder))
            os.makedirs(train_folder, exist_ok=True)
            for file in not_files:
                copy_file_no_tmp(os.path.join(class_path, file), os.path.join(train_folder, file))
        zip_subject(sub_folder, temp_zip_folder)
    final_zip = f"{loso_folder}.zip"
    merge_subject_zips(temp_zip_folder, final_zip)
    shutil.rmtree(temp_zip_folder)
    print(f"最终压缩包生成: {final_zip}")


def delete_main_2():
    casme2_dst_root_path = "/kaggle/working/CASME2_onset_apex_offset_retinaface"
    samm_dst_root_path = "/kaggle/working/SAMM_onset_apex_offset_retinaface"
    casme3_dst_root_path = "/kaggle/working/CASME3_onset_apex_offset_retinaface"
    casme2_output_folder = "/kaggle/working/CASME2_optflow_retinaface"
    samm_output_folder = "/kaggle/working/SAMM_optflow_retinaface"
    casme3_output_folder = "/kaggle/working/CASME3_optflow_retinaface"
    delete_directory(casme2_dst_root_path)
    delete_directory(samm_dst_root_path)
    delete_directory(casme3_dst_root_path)
    delete_directory(casme2_output_folder)
    delete_directory(samm_output_folder)
    delete_directory(casme3_output_folder)


if __name__ == "__main__":
    # delete_main_2()

    # ===================== SMIC LOSO 3分类 =====================
    print("=== 开始处理 SMIC LOSO ===")
    data_folder_3 = '/kaggle/working/SMIC_retinaface_3'
    loso_folder_3 = '/kaggle/working/SMIC_retinaface_loso_3'
    os.makedirs(loso_folder_3, exist_ok=True)

    print_disk_usage()
    process_loso_each(data_folder_3, loso_folder_3, num_classes=3, dataset_name="SMIC")
    zip_frames(loso_folder_3, f'{loso_folder_3}.zip')
    print_directory_structure(loso_folder_3, directory_name='SMIC_retinaface_loso_3')

    # delete_directory(data_folder_3)
    # delete_directory(loso_folder_3)