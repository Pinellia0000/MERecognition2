import cv2
import os
import numpy as np
import pandas as pd
import zipfile
import datetime
from tqdm import tqdm
import shutil


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


def pol2cart(rho, phi):  # Convert polar coordinates to cartesian coordinates for computation of optical strain
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


def computeStrain(u, v):
    u_x = u - pd.DataFrame(u).shift(-1, axis=1)
    v_y = v - pd.DataFrame(v).shift(-1, axis=0)
    u_y = u - pd.DataFrame(u).shift(-1, axis=0)
    v_x = v - pd.DataFrame(v).shift(-1, axis=1)
    os = np.array(np.sqrt(u_x ** 2 + v_y ** 2 + 1 / 2 * (u_y + v_x) ** 2).ffill(axis=1).ffill(axis=0))
    return os


def calculate_optical_flow(img1, img2):
    frame1 = cv2.imread(img1, 0)
    frame2 = cv2.imread(img2, 0)

    optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
    flow = optical_flow.calc(frame1, frame2, None)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    u, v = pol2cart(magnitude, angle)
    os_ = computeStrain(u, v)

    final_u = cv2.resize(u, (48, 48))
    final_v = cv2.resize(v, (48, 48))
    final_os = cv2.resize(os_, (48, 48))

    if ((np.max(final_u) - np.min(final_u)) == 0):
        normalized_u = final_u.astype(np.uint8)
    else:
        normalized_u = ((final_u - np.min(final_u)) / (np.max(final_u) - np.min(final_u)) * 255).astype(np.uint8)

    if ((np.max(final_v) - np.min(final_v)) == 0):
        normalized_v = final_v.astype(np.uint8)
    else:
        normalized_v = ((final_v - np.min(final_v)) / (np.max(final_v) - np.min(final_v)) * 255).astype(np.uint8)

    if ((np.max(final_os) - np.min(final_os)) == 0):
        normalized_os = final_os.astype(np.uint8)
    else:
        normalized_os = ((final_os - np.min(final_os)) / (np.max(final_os) - np.min(final_os)) * 255).astype(np.uint8)

    return normalized_u, normalized_v, normalized_os


def main(input_folder, output_folder):
    for folder_name in tqdm(os.listdir(input_folder), desc="处理视频文件夹"):
        folder_path = os.path.join(input_folder, folder_name)
        out_folder_path = os.path.join(output_folder, folder_name)
        os.makedirs(out_folder_path, exist_ok=True)

        # 获取所有图片
        all_imgs = [img for img in os.listdir(folder_path) if img.endswith(".jpg")]

        # 按前缀分组
        events = {}
        for img_name in all_imgs:
            # 提取事件前缀，例如 sub01_EP01_c
            prefix = "_".join(img_name.split('_')[:-1])
            if prefix not in events:
                events[prefix] = {}
            if img_name.endswith("onset.jpg"):
                events[prefix]['onset'] = img_name
            elif img_name.endswith("apex.jpg"):
                events[prefix]['apex'] = img_name
            elif img_name.endswith("offset.jpg"):
                events[prefix]['offset'] = img_name

        # 遍历每个事件
        for prefix, imgs in events.items():
            # 确保三帧都存在
            if 'onset' not in imgs or 'apex' not in imgs or 'offset' not in imgs:
                print(f"[WARNING] 缺少关键帧: {prefix}")
                print(f"imgs: {imgs}")
                """
                缺少关键帧: 028_4_1 032_3_1
                缺少关键帧: spNO.214_c_5 spNO.149_d_112 spNO.40_e_2327 spNO.40_e_1812
                """
                for img in imgs.values():
                    delete_img_path = os.path.join(folder_path, img)
                    if os.path.exists(delete_img_path):
                        os.remove(delete_img_path)
                        # 删除路径为delete_img_path的图片
                        print(f"已删除路径为: {delete_img_path}的图片")
                continue

            onset_path = os.path.join(folder_path, imgs['onset'])
            apex_path = os.path.join(folder_path, imgs['apex'])
            offset_path = os.path.join(folder_path, imgs['offset'])

            # 计算光流
            flow_1_u, flow_1_v, _ = calculate_optical_flow(onset_path, apex_path)
            flow_2_u, flow_2_v, _ = calculate_optical_flow(apex_path, offset_path)

            # 构造输出文件名
            output_filenames = {
                "1_u": os.path.join(out_folder_path, f"{prefix}_1_u.jpg"),
                "1_v": os.path.join(out_folder_path, f"{prefix}_1_v.jpg"),
                "2_u": os.path.join(out_folder_path, f"{prefix}_2_u.jpg"),
                "2_v": os.path.join(out_folder_path, f"{prefix}_2_v.jpg"),
            }

            # 保存光流图
            cv2.imwrite(output_filenames["1_u"], flow_1_u)
            cv2.imwrite(output_filenames["1_v"], flow_1_v)
            cv2.imwrite(output_filenames["2_u"], flow_2_u)
            cv2.imwrite(output_filenames["2_v"], flow_2_v)


if __name__ == "__main__":
    # CASMEⅡ 数据集
    casme2_input_folder = '/kaggle/working/CASME2_onset_apex_offset_retinaface'
    casme2_output_folder = "/kaggle/working/CASME2_optflow_retinaface"
    main(casme2_input_folder, casme2_output_folder)
    zipPath = '/kaggle/working/CASME2_optflow_retinaface.zip'
    zip_frames(casme2_output_folder, zipPath)
    print_directory_structure(casme2_output_folder, directory_name='CASME2_optflow_retinaface')


    # SAMM 数据集
    samm_input_folder = '/kaggle/working/SAMM_onset_apex_offset_retinaface'
    samm_output_folder = "/kaggle/working/SAMM_optflow_retinaface"
    main(samm_input_folder, samm_output_folder)
    zipPath = '/kaggle/working/SAMM_optflow_retinaface.zip'
    zip_frames(samm_output_folder, zipPath)
    print_directory_structure(samm_output_folder, directory_name='SAMM_optflow_retinaface')


    # CAS(ME)^3 数据集
    casme3_input_folder = '/kaggle/working/CASME3_onset_apex_offset_retinaface'
    casme3_output_folder = "/kaggle/working/CASME3_optflow_retinaface"
    main(casme3_input_folder, casme3_output_folder)
    zipPath = '/kaggle/working/CASME3_optflow_retinaface.zip'
    zip_frames(casme3_output_folder, zipPath)
    print_directory_structure(casme3_output_folder, directory_name='CASME3_optflow_retinaface')

