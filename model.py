import os
os.environ['USE_PATH_FOR_GDAL_PYTHON']='YES'

import onnxruntime as rt
import numpy as np
from osgeo import gdal,osr
import cv2
import geopandas as gdb
from shapely.geometry import shape
import argparse
import json
import ast
from tqdm import tqdm
from loguru import logger
# providers = ['CPUExecutionProvider']
# providers = ['CUDAExecutionProvider']

def ReadDataBy_Rect(filename, bounds, res=None):
    '''
    按照指定经纬度坐标范围读取数据
    :param filename:  文件名 ,可以为gdal能读取的任何格式，包括vrt
    :param bounds:  经纬度坐标范围，[xmin,ymin,xmax,ymax]
    :param res: 读取数据的分辨率
    '''
    options = gdal.WarpOptions(outputBoundsSRS='EPSG:4326', outputBounds=bounds, xRes=res, yRes=res, format='VRT')
    ds = gdal.Dataset = gdal.Warp('', filename, options=options)
    geotran = ds.GetGeoTransform()
    proj = ds.GetProjection()
    data = ds.ReadAsArray().transpose(1, 2, 0)
    ds = None
    return geotran, proj, data


def calculate_cut_range(img_size, patch_size, overlap=0.5, pad_edge=1):
    patch_range = []
    patch_height = patch_size[0]
    patch_width = patch_size[1]
    width_overlap = int(patch_width * (1-overlap))
    height_overlap = int(patch_height * (1-overlap))

    for x_s in range(0,img_size[0],width_overlap):
        x_e = min(x_s+patch_width,img_size[0])
        for y_s in range(0,img_size[1],height_overlap):
            y_e = min(y_s+patch_height,img_size[1])
            patch_range.append([int(y_s), int(y_e), int(x_s), int(x_e)])

    return patch_range


def calculate_cut_range2(img_size, patch_size, overlap=0.5, pad_edge=1):
    patch_range = []
    patch_height = patch_size[0]
    patch_width = patch_size[1]
    width_overlap = patch_width * overlap
    height_overlap = patch_height * overlap
    cols = img_size[1]
    rows = img_size[0]
    x_e = 0
    while (x_e < cols - 1):
        y_e = 0
        x_s = max(0, x_e - width_overlap)
        x_e = x_s + patch_width
        if (x_e > cols):
            x_e = cols - 1
        if (pad_edge == 1):  ## if the last path is not enough, then extent to the inerside.
            x_s = x_e - patch_width
            x_s = max(0, x_s)
        if (pad_edge == 2):  ## if the last patch is not enough, then extent to the outside(with black).
            x_s = x_s
        while (y_e < rows - 1):
            y_s = max(0, y_e - height_overlap)
            y_e = y_s + patch_height
            if (y_e > rows):
                y_e = rows - 1
            if (pad_edge == 1):  ## if the last path is not enough, then extent to the inerside.
                y_s = y_e - patch_height
                y_s = max(0, y_s)
            if (pad_edge == 2):  ## if the last patch is not enough, then extent to the outside(with black).
                y_s = y_s
            patch_range.append([int(y_s), int(y_e), int(x_s), int(x_e)])
    return patch_range


def mask2geojson(mask, geotrans, proj, CLASSES, cls, save_path):
    x_geo_start = geotrans[0]
    x_cell_size = geotrans[1]
    y_geo_start = geotrans[3]
    y_cell_size = geotrans[5]
    features = []
    for category in cls:
        binary_mask = cv2.inRange(mask, category, category)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary_mask = cv2.dilate(binary_mask, kernel, iterations=2)
        binary_mask = cv2.erode(binary_mask, kernel, iterations=2)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if len(contour) >= 4:
                contour = contour.squeeze()
                contour[:, 0] = contour[:, 0] * x_cell_size + x_geo_start
                contour[:, 1] = contour[:, 1] * y_cell_size + y_geo_start

                geometry = {
                    "type": "Polygon",
                    "coordinates": [contour]
                }
                feature = {
                    "type": "Feature",
                    "properties": {"category": CLASSES[category],"category_value":category},
                    "geometry": shape(geometry)
                }
                features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features

    }

    with open(save_path, 'w') as f:
        json_txt = gdb.GeoDataFrame.from_features(geojson).to_json()
        json_obj = json.loads(json_txt)
        json_obj['crs'] ={ "type": "name", "properties": { "name": osr.SpatialReference(proj).ExportToProj4()} }

        # f.write(json.dumps(geojson))
        f.write(json.dumps(json_obj))


def getproj(proj):
    return osr.SpatialReference(proj).ExportToProj4()

def predict(model_path, input_path, output_path, CLASSES, desired_classes, region, providers, process_size=[1024, 1024], res=0.2, overlap=0.2):
    geotrans, proj, data = ReadDataBy_Rect(input_path, region, res=res)
    width, height = data.shape[0], data.shape[1]

    predict_data = np.zeros((width, height))
    patch_ranges = calculate_cut_range([height, width], patch_size=process_size, overlap=overlap, pad_edge=1)
    # patch_ranges = calculate_cut_range([width, height], patch_size=process_size, overlap=overlap, pad_edge=1)

    # for ymin,ymax,xmin,xmax in patch_ranges:
    #     logger.info(f'height:{ymax-ymin}--width:{xmax-xmin}')

    m = rt.InferenceSession(model_path, providers=providers)
    # for inds in range(len(patch_ranges)):
    # pbar = tqdm(total=len(patch_ranges))
    inds = 0
    for y_s,y_e,x_s,x_e in patch_ranges:
        # logger.info(inds)
        # y_s = round(patch_ranges[inds][0])
        # y_e = round(patch_ranges[inds][1])
        # x_s = round(patch_ranges[inds][2])
        # x_e = round(patch_ranges[inds][3])
        data_patch = data[y_s:y_e, x_s:x_e, :]
        if data_patch.shape[0] == data_patch.shape[1] == process_size[0]:
            data_input = np.expand_dims(np.transpose((np.array(data_patch, np.float32)) / 255, (2, 0, 1)), 0)
            # logger.info(data_input.shape)
            param = {'input': data_input}
            # m = rt.InferenceSession(model_path, providers=providers)
            r = m.run(None, param)
        else:
            # continue
            p_width, p_height = data_patch.shape[0], data_patch.shape[1]
            pad_height = (32 - (p_height % 32)) % 32
            pad_width = (32 - (p_width % 32)) % 32
            p_data_patch = np.pad(data_patch, ((0, pad_width), (0, pad_height), (0, 0)), mode='constant')
            p_data_input = np.expand_dims(np.transpose((np.array(p_data_patch, np.float32)) / 255, (2, 0, 1)), 0)
            # logger.info(p_data_input.shape)
            param = {'input': p_data_input}
            # m = rt.InferenceSession(model_path, providers=providers)
            r = m.run(None, param)
            r = [arr[:, :, :p_width, :p_height] for arr in r]

        pr = r[0].transpose(0, 2, 3, 1)[0]
        pr = pr.argmax(axis=-1)
        valid_mask = np.isin(pr, desired_classes)
        modify_pr = np.where(valid_mask, pr, 0)
        if y_s == 0:
            roi_r_s = 0
        else:
            roi_r_s = round(y_s + process_size[0] * overlap / 2)
        if y_e == height - 1:
            roi_r_e = y_e
        else:
            roi_r_e = round(y_e - process_size[0] * overlap / 2)

        if x_s == 0:
            roi_c_s = 0
        else:
            roi_c_s = round(x_s + process_size[0] * overlap / 2)
        if x_e == width - 1:
            roi_c_e = x_e
        else:
            roi_c_e = round(x_e - process_size[0] * overlap / 2)

        modify_pr2 = modify_pr[roi_r_s - y_s:roi_r_e - y_s, roi_c_s - x_s:roi_c_e - x_s]
        predict_data[roi_r_s:roi_r_e, roi_c_s:roi_c_e] = modify_pr2
        inds += 1
        logger.info(f'完成:{inds}/{len(patch_ranges)}')
        # break

    logger.info('开始生成矢量数据')
    mask2geojson(predict_data, geotrans, proj, CLASSES, desired_classes, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_path', help='onnx模型路径')
    parser.add_argument('-input_path', help='输入路径，vrt或tif')
    parser.add_argument('-output_path', help='输出路径，geojson')
    parser.add_argument('-classes', help='类别字典')
    parser.add_argument('-categorys', type=int, nargs='+', help='需要预测的类别')
    parser.add_argument('-region', type=float, nargs='+', help='预测范围，经纬度坐标')
    parser.add_argument('-res', type=float, help='读取影像分辨率')
    parser.add_argument('-providers', help='cpu,cuda')
    args = parser.parse_args()
    res = None
    if args.res:
        res = res
    logger.info(f'model_path:{args.model_path}')
    logger.info(f'input_path:{args.input_path}')
    logger.info(f'output_path:{args.output_path}')
    logger.info(f'classes:{args.classes}')
    logger.info(f'categorys:{args.categorys}')
    logger.info(f'region:{args.region}')
    logger.info(f'res:{res}')
    logger.info(f'provider:{args.providers}')
    # exit()

    CLASSES = ast.literal_eval(args.classes)
    if type(CLASSES) is str:
        CLASSES = ast.literal_eval(args.classes[1:-1])
    model_path = args.model_path
    region = args.region
    input_path = args.input_path
    output_path = args.output_path
    output_path_dir = os.path.dirname(output_path)
    if not os.path.exists(output_path_dir):
        os.makedirs(output_path_dir)

    providers = args.providers
    if providers == 'cpu' or providers == 'CPU':
        providers = ['CPUExecutionProvider']
    elif providers == 'cuda' or providers == 'CUDA':
        providers = ['CUDAExecutionProvider']
    desired_classes = args.categorys

    predict(model_path, input_path, output_path, CLASSES, desired_classes, region, providers, res=0.2)

    logger.info('预测完成。')