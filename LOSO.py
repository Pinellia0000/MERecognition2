import os
import shutil
import zipfile
import datetime
from tqdm import tqdm
from collections import defaultdict


def zip_subject(subject_folder, temp_zip_folder):
    """
    压缩单个 subject 文件夹到临时 zip
    """
    os.makedirs(temp_zip_folder, exist_ok=True)
    subject_name = os.path.basename(subject_folder)
    zip_path = os.path.join(temp_zip_folder, f"{subject_name}.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as zipf:
        for root, dirs, files in os.walk(subject_folder):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, subject_folder)
                zipf.write(file_path, arcname)
    # 压缩完成后删除原始 subject 文件夹，节省空间
    shutil.rmtree(subject_folder)
    return zip_path


def merge_subject_zips(temp_zip_folder, final_zip):
    """
    合并所有 subject 的临时 zip 到最终 zip
    """
    zip_files = sorted(os.listdir(temp_zip_folder))
    with zipfile.ZipFile(final_zip, 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as final_zipf:
        for zf_name in zip_files:
            zf_path = os.path.join(temp_zip_folder, zf_name)
            with zipfile.ZipFile(zf_path, 'r') as sub_zip:
                for file in sub_zip.namelist():
                    data = sub_zip.read(file)
                    # 在总 zip 中加上 subject 名称前缀，避免冲突
                    arcname = os.path.join(zf_name.replace(".zip", ""), file).replace("\\", "/")
                    final_zipf.writestr(arcname, data)
            os.remove(zf_path)  # 删除临时 zip


def copy_file_no_tmp(src, dst, buffer_size=1024 * 1024):
    """
    CAS(ME)^2数据集用
    """
    try:
        with open(src, 'rb') as fsrc, open(dst, 'wb') as fdst:
            while True:
                buf = fsrc.read(buffer_size)
                if not buf:
                    break
                fdst.write(buf)
    except OSError as e:
        if e.errno == 28:  # 磁盘满
            print(f"[ERROR] 磁盘空间不足，拷贝失败: {src}")
            print_disk_usage("/kaggle/working")
            raise
        else:
            raise


def print_disk_usage(path="/kaggle/working"):
    """
    输出指定路径的磁盘总容量、已用容量和可用容量（单位GB）
    """
    usage = shutil.disk_usage(path)
    total_gb = usage.total / (1024 ** 3)
    used_gb = usage.used / (1024 ** 3)
    free_gb = usage.free / (1024 ** 3)

    print(f"磁盘路径: {path}")
    print(f"总容量: {total_gb:.2f} GB")
    print(f"已用: {used_gb:.2f} GB")
    print(f"可用: {free_gb:.2f} GB")


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


def zip_frames_stream(packagePath, zipPath):
    """
    CAS(ME)^2使用
    流式压缩目录，避免一次性占用大量内存。
    packagePath: 要压缩的目录
    zipPath: 压缩包路径
    """
    if os.path.exists(zipPath):
        os.remove(zipPath)

    with zipfile.ZipFile(zipPath, 'w', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zipf:
        for root, dirs, files in os.walk(packagePath):
            for file in files:
                file_path = os.path.join(root, file)
                # 计算在 zip 包里的相对路径
                arcname = os.path.relpath(file_path, packagePath).replace("\\", "/")
                zipf.write(file_path, arcname)
    print("打包完成")
    print(datetime.datetime.utcnow())


def print_zip_structure(zip_path):
    """
    打印 zip 压缩包内部目录结构（仿照 print_directory_structure 风格）
    """
    print(f"{zip_path} 压缩包内容结构如下：\n")

    with zipfile.ZipFile(zip_path, 'r') as zipf:
        all_files = sorted(zipf.namelist())

        # 构建一个简单树状字典
        tree = lambda: defaultdict(tree)
        root = tree()

        for f in all_files:
            parts = f.strip("/").split("/")
            node = root
            for part in parts:
                node = node[part]

        # 递归打印
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


def process_loso_each(data_folder, loso_folder, num_classes, dataset_name="Dataset"):
    """
    每次只处理一个数据集的一种分类
    data_folder: 已分类的数据集 (例如 CASME2_retinaface_5)
    loso_folder: 输出路径 (例如 CASME2_retinaface_loso_5)
    num_classes: 分类数 (5 或 3 或 7 等)
    dataset_name: 用于 tqdm 的提示信息
    """

    # 提取所有被试前缀
    subjects = set()
    for class_folder in range(num_classes):
        class_path = os.path.join(data_folder, str(class_folder))
        if not os.path.exists(class_path):
            continue
        for file in os.listdir(class_path):
            # CASMEⅡ SAMM  CAS(ME)^2 都是取文件名下划线分隔的第一个
            # CASMEⅡ sub01_EP19_05f_apex
            # SAMM 006_1_2_apex
            # CAS(ME)^2 spNO.1_a_355_apex
            if file.startswith("sub") or file.startswith("spNO"):
                subjects.add(file.split("_")[0])
            else:
                subjects.add(file.split("_")[0].zfill(3))
    subjects = sorted(subjects)

    # tqdm 进度条
    for subject in tqdm(subjects, desc=f"Processing {dataset_name} subjects"):
        sub_folder = os.path.join(loso_folder, subject)
        os.makedirs(sub_folder, exist_ok=True)

        for class_folder in range(num_classes):
            class_path = os.path.join(data_folder, str(class_folder))
            if not os.path.exists(class_path):
                continue

            files = [file for file in os.listdir(class_path) if file.startswith(subject)]
            not_files = [file for file in os.listdir(class_path) if not file.startswith(subject)]

            # 测试集
            if files:
                test_folder = os.path.join(sub_folder, 'test', str(class_folder))
                os.makedirs(test_folder, exist_ok=True)
                for file in files:
                    shutil.copy(os.path.join(class_path, file), os.path.join(test_folder, file))

            # 训练集
            train_folder = os.path.join(sub_folder, 'train', str(class_folder))
            os.makedirs(train_folder, exist_ok=True)
            for file in not_files:
                shutil.copy(os.path.join(class_path, file), os.path.join(train_folder, file))


def process_loso_each_CASME3(data_folder, loso_folder, num_classes, dataset_name="Dataset"):
    """
    专门处理CAS(ME)^3
    data_folder: 已分类的数据集 (例如 CASME2_retinaface_5)
    loso_folder: 输出路径 (例如 CASME2_retinaface_loso_5)
    num_classes: 分类数 (5 或 3 或 7 等)
    dataset_name: 用于 tqdm 的提示信息
    """

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

        # 拷贝文件到 subject 文件夹
        for class_folder in range(num_classes):
            class_path = os.path.join(data_folder, str(class_folder))
            if not os.path.exists(class_path):
                continue

            files = [file for file in os.listdir(class_path) if file.startswith(subject)]
            not_files = [file for file in os.listdir(class_path) if not file.startswith(subject)]

            # 测试集
            if files:
                test_folder = os.path.join(sub_folder, 'test', str(class_folder))
                os.makedirs(test_folder, exist_ok=True)
                for file in files:
                    copy_file_no_tmp(os.path.join(class_path, file), os.path.join(test_folder, file))

            # 训练集
            train_folder = os.path.join(sub_folder, 'train', str(class_folder))
            os.makedirs(train_folder, exist_ok=True)
            for file in not_files:
                copy_file_no_tmp(os.path.join(class_path, file), os.path.join(train_folder, file))

        # 完成该 subject 后立即压缩
        zip_subject(sub_folder, temp_zip_folder)

    # 合并所有 subject zip
    final_zip = f"{loso_folder}.zip"
    merge_subject_zips(temp_zip_folder, final_zip)
    # 删除临时文件夹
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
    # # 减少一些目录
    # delete_main_2()
    # CASMEⅡ 数据集
    data_folder_5 = '/kaggle/working/CASME2_retinaface_5'  # 原始数据路径
    loso_folder_5 = '/kaggle/working/CASME2_retinaface_loso_5'  # 新路径
    data_folder_3 = '/kaggle/working/CASME2_retinaface_3'  # 原始数据路径
    loso_folder_3 = '/kaggle/working/CASME2_retinaface_loso_3'  # 新路径
    os.makedirs(loso_folder_5, exist_ok=True)
    os.makedirs(loso_folder_3, exist_ok=True)

    # 输出磁盘容量
    print_disk_usage()
    process_loso_each(data_folder_5, loso_folder_5, num_classes=5, dataset_name="CASMEⅡ")
    zipPath = f'{loso_folder_5}.zip'
    zip_frames(loso_folder_5, zipPath)
    print_directory_structure(loso_folder_5, directory_name='CASME2_retinaface_loso_5')
    # 减少一部分目录
    delete_directory(data_folder_5)
    delete_directory(loso_folder_5)

    # 输出磁盘容量
    print_disk_usage()
    process_loso_each(data_folder_3, loso_folder_3, num_classes=3, dataset_name="CASMEⅡ")
    zipPath = f'{loso_folder_3}.zip'
    zip_frames(loso_folder_3, zipPath)
    print_directory_structure(loso_folder_3, directory_name='CASME2_retinaface_loso_3')
    delete_directory(data_folder_3)
    delete_directory(loso_folder_3)

    # # SAMM 数据集
    # data_folder_3 = '/kaggle/working/SAMM_retinaface_3'  # 原始数据路径
    # loso_folder_3 = '/kaggle/working/SAMM_retinaface_loso_3'  # 新路径
    # os.makedirs(loso_folder_3, exist_ok=True)
    #
    # # 输出磁盘容量
    # print_disk_usage()
    # process_loso_each(data_folder_3, loso_folder_3, num_classes=3, dataset_name="SAMM")
    # zipPath = f'{loso_folder_3}.zip'
    # zip_frames(loso_folder_3, zipPath)
    # print_directory_structure(loso_folder_3, directory_name='SAMM_retinaface_loso_3')
    # delete_directory(data_folder_3)
    # delete_directory(loso_folder_3)
    #
    # # CAS(ME)^3 数据集
    # data_folder_7 = '/kaggle/working/CASME3_retinaface_7'  # 原始数据路径
    # loso_folder_7 = '/kaggle/working/CASME3_retinaface_loso_7'  # 新路径
    # data_folder_4 = '/kaggle/working/CASME3_retinaface_4'  # 原始数据路径
    # loso_folder_4 = '/kaggle/working/CASME3_retinaface_loso_4'  # 新路径
    # data_folder_3 = '/kaggle/working/CASME3_retinaface_3'  # 原始数据路径
    # loso_folder_3 = '/kaggle/working/CASME3_retinaface_loso_3'  # 新路径
    # os.makedirs(loso_folder_7, exist_ok=True)
    # os.makedirs(loso_folder_4, exist_ok=True)
    # os.makedirs(loso_folder_3, exist_ok=True)
    #
    # # 输出磁盘容量
    # print_disk_usage()
    # process_loso_each_CASME3(data_folder_7, loso_folder_7, num_classes=7, dataset_name="CAS(ME)^3")
    # zipPath = f'{loso_folder_7}.zip'
    # # print_zip_structure(zipPath)
    # delete_directory(data_folder_7)
    # delete_directory(loso_folder_7)
    #
    # # 输出磁盘容量
    # print_disk_usage()
    # process_loso_each_CASME3(data_folder_4, loso_folder_4, num_classes=4, dataset_name="CAS(ME)^3")
    # zipPath = f'{loso_folder_4}.zip'
    # # print_zip_structure(zipPath)
    # delete_directory(data_folder_4)
    # delete_directory(loso_folder_4)
    #
    # # 输出磁盘容量
    # print_disk_usage()
    # process_loso_each_CASME3(data_folder_3, loso_folder_3, num_classes=3, dataset_name="CAS(ME)^3")
    # zipPath = f'{loso_folder_3}.zip'
    # # print_zip_structure(zipPath)
    # delete_directory(data_folder_3)
    # delete_directory(loso_folder_3)
