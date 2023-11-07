import argparse
import logging
import os

os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = pow(2, 40).__str__()
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from nets.unet import Unet as unet
from osgeo import gdal
from pathlib import Path
import glob
from scipy.ndimage import morphology
def calculate_cut_range(img_size, patch_size,overlap=0.5,pad_edge=1):
    patch_range = []
    patch_height = patch_size[0]
    patch_width = patch_size[1]
    width_overlap = patch_width * overlap
    height_overlap = patch_height * overlap
    cols=img_size[1]
    rows=img_size[0]
    x_e = 0
    while (x_e < cols-1):
        y_e=0
        x_s = max(0, x_e - width_overlap)
        x_e = x_s + patch_width
        if (x_e > cols):
            x_e = cols-1
        if (pad_edge == 1): ## if the last path is not enough, then extent to the inerside.
            x_s = x_e - patch_width
            x_s = max(0,x_s)
        if (pad_edge == 2):## if the last patch is not enough, then extent to the outside(with black).
            x_s=x_s
        while (y_e < rows-1):
            y_s = max(0, y_e - height_overlap)
            y_e = y_s + patch_height
            if (y_e > rows):
                y_e = rows-1
            if (pad_edge == 1): ## if the last path is not enough, then extent to the inerside.
                y_s = y_e - patch_height
                y_s = max(0, y_s)
            if (pad_edge == 2):## if the last patch is not enough, then extent to the outside(with black).
                y_s=y_s
            patch_range.append([int(y_s),int(y_e),int(x_s),int(x_e)])
    return patch_range

def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image
def read_img(filename):
    dataset = gdal.Open(filename)  # 打开文件

    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数

    im_geotrans = dataset.GetGeoTransform()  # 获取地理参考信息
    im_proj = dataset.GetProjection()  # 地图投影信息
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 将数据写成数组，对应栅格矩阵

    del dataset
    return im_proj, im_geotrans, im_data

def load_img(imgPath):
    """
    Load image
    :param imgPath: path of the image to load
    :return: numpy array of the image
    """
    im_proj = []
    im_geotrans = []
    if imgPath.endswith('.tif'):
        im_proj, im_geotrans, im_data = read_img(imgPath)
    else:
        img = np.array(cv2.imread(imgPath))
        im_data = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if im_data.shape[0] < 10:
        im_data = np.transpose(im_data, (1, 2, 0))
    return im_proj, im_geotrans, im_data

def write_img(filename, im_proj='', im_geotrans='', im_data=[]):

    # 判断栅格数据的数据类型
    if 'uint8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    # 判读数组维数
    if len(im_data.shape) == 3:
        im_height, im_width, im_bands = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape

        # 创建文件
    driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)
    if im_geotrans:
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
    if im_proj:
        dataset.SetProjection(im_proj)  # 写入投影

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[:, :, i])
    del dataset

def predict_img(net, full_img, device):

    data = np.expand_dims((np.array(full_img, np.float32)) / 255, 0)
    with torch.no_grad():
        images = torch.from_numpy(data).to(device=device, dtype=torch.float32)
        output = net(images)[0]
        pr = F.softmax(output.permute(1, 2, 0), dim=-1).cpu().numpy()
    return pr

def load_models(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net_rgb = unet(num_classes=6, backbone="resnet50")
    net_rgb.to(device=device)
    dom_weight_path = Path(model_path)
    net_rgb.load_state_dict(torch.load(dom_weight_path, map_location=device)['model_state_dict'])
    return net_rgb

def FindAllInput(dom_folder, mask_out_folder):
    data_files = []
    mask_files = []
    dom_files = glob.glob(os.path.join(dom_folder, '*.tif'))
    for dom_file in dom_files:
        pathes = dom_file.split(os.sep)
        name = pathes[-1]
        mask_file = os.path.join(mask_out_folder, name)
        data_files.append(dom_file)
        mask_files.append(mask_file)
    return data_files, mask_files

def oversize_tif_predict(imgPath, out_path, net, device, class_list, process_size=[1024, 1024], overlap=0.2, num_class=6):

    dom_data = gdal.Open(imgPath)  # 打开文件
    width = dom_data.RasterXSize  # 栅格矩阵的列数
    height = dom_data.RasterYSize  # 栅格矩阵的行数

    out_label = dom_data.GetDriver().Create(out_path, width, height, 1, gdal.GDT_Byte)
    out_label.SetProjection(dom_data.GetProjection())
    out_label.SetGeoTransform(dom_data.GetGeoTransform())
    # overlap=0.2
    patch_ranges = calculate_cut_range([height, width], patch_size=process_size, overlap=overlap, pad_edge=1)

    ### load tif data by patch
    for inds in range(len(patch_ranges)):
        with tqdm(total=len(patch_ranges), desc=f'patch {inds}/{len(patch_ranges)}', unit='img') as pbar:
            y_s = round(patch_ranges[inds][0])
            y_e = round(patch_ranges[inds][1])
            x_s = round(patch_ranges[inds][2])
            x_e = round(patch_ranges[inds][3])
            dom_patch = np.empty([y_e - y_s, x_e - x_s, 3], dtype=np.uint8)
            for i in range(1, 4):
                band = dom_data.GetRasterBand(i)
                data1 = band.ReadAsArray(int(x_s), int(y_s), int(x_e) - int(x_s), int(y_e) - int(y_s)).astype(np.uint8)
                dom_patch[:, :, i - 1] = data1
            dom_patch = np.transpose(dom_patch, (2, 0, 1))
            input_data = dom_patch

            mask = predict_img(net=net, full_img=input_data, device=device)
            mask = np.argmax(mask, axis=-1)

            # # 形态学滤波
            # mask_array = np.zeros_like(mask)
            # for index in range(1, num_class+1):
            #     binary_mask = cv2.inRange(mask, index, index)
            #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            #     binary_mask = cv2.erode(binary_mask, kernel, iterations=2)
            #
            #     binary_mask = cv2.dilate(binary_mask, kernel, iterations=2)
            #     mask_array[binary_mask > 0] = index
            #
            # mask2 = np.isin(mask_array, class_list)
            # mask3 = np.where(mask2, mask_array, 0)

            mask2 = np.isin(mask, class_list)
            mask3 = np.where(mask2, mask, 0)

            if y_s == 0:
                roi_r_s = 0
            else:
                # roi_r_s = round(y_s + process_size[0]/4)
                roi_r_s = round(y_s + process_size[0] * overlap / 2)
            if y_e == height - 1:
                roi_r_e = y_e
            else:
                # roi_r_e = round(y_e - process_size[0]/4)
                roi_r_e = round(y_e - process_size[0] * overlap / 2)

            if x_s == 0:
                roi_c_s = 0
            else:
                # roi_c_s = round(x_s + process_size[1]/4)
                roi_c_s = round(x_s + process_size[0] * overlap / 2)
            if x_e == width - 1:
                roi_c_e = x_e
            else:
                # roi_c_e = round(x_e - process_size[0]/4)
                roi_c_e = round(x_e - process_size[0] * overlap / 2)

            roi_data = mask3[roi_r_s - y_s:roi_r_e - y_s, roi_c_s - x_s:roi_c_e - x_s]
            out_label.GetRasterBand(1).WriteArray(roi_data, int(roi_c_s), int(roi_r_s))
            out_label.FlushCache()

    del out_label
def NormalSizePredict(filename, mask_file, net, device, class_list):
    out_filename = mask_file

    im_proj, im_geotrans, extra_data = load_img(filename)
    org_shap = [extra_data.shape[0], extra_data.shape[1]]

    out_label_temp = np.zeros((org_shap[0], org_shap[1], 6), dtype=np.float32)
    rows = org_shap[0]
    cols = org_shap[1]

    path_size = [2048, 2048]

    overlap_ratio = 0.6
    patch_ranges = calculate_cut_range([rows, cols], patch_size=path_size, overlap=overlap_ratio)
    for inds in range(len(patch_ranges)):
        with tqdm(total=len(patch_ranges), desc=f'patch {inds}/{len(patch_ranges)}', unit='img') as pbar:
            y_s = round(patch_ranges[inds][0])
            y_e = round(patch_ranges[inds][1])
            x_s = round(patch_ranges[inds][2])
            x_e = round(patch_ranges[inds][3])
            img_patch = extra_data[int(y_s):int(y_e), int(x_s):int(x_e)]
            image_data = np.transpose(img_patch, (2, 0, 1))
            input_data = image_data
            mask = predict_img(net=net, full_img=input_data, device=device)
            out_label_temp[int(y_s):int(y_e), int(x_s):int(x_e), :] = out_label_temp[int(y_s):int(y_e),int(x_s):int(x_e), :] + mask

    out_label = np.argmax(out_label_temp, axis=-1)
    out_label2 = np.isin(out_label, class_list)
    out_label = np.where(out_label2, out_label, 0)

    result = (out_label).astype(np.uint8)
    write_img(out_filename, im_proj, im_geotrans, result)
    logging.info(f'Mask saved to {out_filename}')

def Predict(input_folder, mask_out_folder, class_list, model_path):

    if os.path.exists(mask_out_folder) == 0:
        os.mkdir(mask_out_folder)

    data_files, mask_files = FindAllInput(input_folder, mask_out_folder)
    net = load_models(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i, filename in enumerate(data_files):
        with tqdm(total=len(data_files), desc=f'patch {i}/{len(data_files)}', unit='img') as pbar:
            logging.info(f'\nPredicting image {filename} ...')
            dataset = gdal.Open(filename)  # 打开文件
            im_width = dataset.RasterXSize  # 栅格矩阵的列数
            im_height = dataset.RasterYSize  # 栅格矩阵的行数

            if im_width > 5000 or im_height > 5000:
                oversize_tif_predict(filename, mask_files[i], net, device, class_list)
            else:
                NormalSizePredict(filename, mask_files[i], net, device, class_list)

def mask2color(input_folder, mask_folder, color_map):
    for filename in os.listdir(input_folder):
        if filename.endswith(".tif"):  # Assuming the mask images are in PNG format
            mask_path = os.path.join(mask_folder, filename)
            img_path = os.path.join(input_folder, filename)
            # Read the mask image
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.imread(img_path)
            # Map mask values to colors
            colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            for mask_value, color in color_map.items():
                colored_mask[mask == mask_value] = color

            # Save the colored mask as a color image
            colored_mask_rgb = cv2.cvtColor(colored_mask, cv2.COLOR_BGR2RGB)

            # blend_img = cv2.addWeighted(img, 0.5, colored_mask_rgb, 0.5, 0.0)
            # cv2.imwrite(mask_path, blend_img)
            cv2.imwrite(mask_path, colored_mask_rgb)

if __name__ == '__main__':

    class_list = [1, 2, 3, 4, 5]                # 需要预测类别
    input_folder = r'E:\data\test'              # img输入路径
    mask_out_folder = r'E:\data\predict_result' # mask保存路径
    model_path = './pth/dyh2.pth' # mask保存路径

    color_map = {
        0: [0, 0, 0],
        1: [255, 0, 0],
        2: [0, 255, 0],
        3: [0, 0, 255],
        4: [255, 255, 0],
        5: [255, 0, 255],
    }

    Predict(input_folder, mask_out_folder, class_list, model_path)
    mask2color(input_folder, mask_out_folder, color_map)


