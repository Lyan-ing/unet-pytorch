import os
import sys
import ast
os.environ['USE_PATH_FOR_GDAL_PYTHON'] = 'YES'
import onnxruntime as rt
import numpy as np
from osgeo import gdal, osr
import cv2
import geopandas as gpd
import argparse
import json
from tqdm import tqdm

providers = ['CPUExecutionProvider']
providers = ['CUDAExecutionProvider']
CLASSES = {
    0: 'background',
    1: 'building',
    2: 'glass',
    3: 'tree',
    4: 'water',
    5: 'tea'
}

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

def calculate_cut_range1(img_size, block_size, overlap=0):
    patch_range = []
    patch_height = block_size[0]
    patch_width = block_size[1]
    width_overlap = int(patch_width * (1 - overlap))
    height_overlap = int(patch_height * (1 - overlap))
    w_overlap = int(patch_width * overlap)
    h_overlap = int(patch_height * overlap)

    for x_s in range(0, img_size[1], width_overlap):
        x_e = min(x_s + patch_width, img_size[1])
        # if x_e - x_s < 1024:
        #     x_s = max(0, x_e - 1024)
        for y_s in range(0, img_size[0], height_overlap):
            y_e = min(y_s + patch_height, img_size[0])
            # if y_e - y_s < 1024:
            #     y_s = max(0, y_e - 1024)
            patch_range.append([int(y_s), int(y_e), int(x_s), int(x_e)])

    return patch_range

def calculate_cut_range2(img_size, patch_size, overlap=0.5, pad_edge=1):
    patch_range = []
    patch_height = patch_size[0]
    patch_width = patch_size[1]
    width_overlap = int(patch_width * (1 - overlap))
    height_overlap = int(patch_height * (1 - overlap))
    w_overlap = int(patch_width * overlap)
    h_overlap = int(patch_height * overlap)

    for x_s in range(0, img_size[1], width_overlap):
        if x_s + w_overlap < img_size[1]:
            x_e = min(x_s + patch_width, img_size[1])
            if x_e == img_size[1]:
                x_s = max(0, x_e - patch_width)
        else:
            x_e = img_size[1]
        for y_s in range(0, img_size[0], height_overlap):
            if y_s + h_overlap < img_size[0]:
                y_e = min(y_s + patch_height, img_size[0])
                if y_e == img_size[0]:
                    y_s = max(0, y_e - patch_height)
            else:
                y_e = img_size[0]

            patch_range.append([int(y_s), int(y_e), int(x_s), int(x_e)])
    return patch_range

def getproj(proj):
    return osr.SpatialReference(proj).ExportToProj4()

def predict(model_path, input_path, output_path, desired_classes, region, process_size=[1024, 1024], overlap=0.2):

    options = gdal.WarpOptions(outputBoundsSRS='EPSG:4326', outputBounds=region, xRes=0.2, yRes=0.2, format='VRT')
    ds = gdal.Dataset = gdal.Warp('', input_path, options=options)
    if ds is None:
        print('输入范围有误')
        sys.exit()

    all_width = ds.RasterXSize
    all_height = ds.RasterYSize
    geotrans = ds.GetGeoTransform()
    x_geo_start, x_cell_size, _, y_geo_start, _, y_cell_size = geotrans
    proj = ds.GetProjection()
    big_patch_ranges = calculate_cut_range1([all_height, all_width], block_size=[10240, 10240], overlap=0)
    m = rt.InferenceSession(model_path, providers=providers)

    all_features = []
    pbar1 = tqdm(total=len(big_patch_ranges))
    for b_y_s, b_y_e, b_x_s, b_x_e in big_patch_ranges:
        block_data = ds.ReadAsArray(b_x_s, b_y_s, b_x_e-b_x_s, b_y_e-b_y_s).transpose(1, 2, 0)
        width, height = block_data.shape[1], block_data.shape[0]
        predict_data = np.zeros((height, width))
        patch_ranges = calculate_cut_range2([height, width], patch_size=process_size, overlap=overlap, pad_edge=1)

        pbar2 = tqdm(total=len(patch_ranges))
        for y_s, y_e, x_s, x_e in patch_ranges:
            data_patch = block_data[y_s:y_e, x_s:x_e, :]
            if not np.all(data_patch == 0):
                if data_patch.shape[0] == data_patch.shape[1] == process_size[0]:
                    data_input = np.expand_dims(np.transpose((np.array(data_patch, np.float32)) / 255, (2, 0, 1)), 0)
                    param = {'input': data_input}
                    r = m.run(None, param)
                else:
                    p_width, p_height = data_patch.shape[0], data_patch.shape[1]
                    pad_height = (32 - (p_height % 32)) % 32
                    pad_width = (32 - (p_width % 32)) % 32
                    p_data_patch = np.pad(data_patch, ((0, pad_width), (0, pad_height), (0, 0)), mode='constant')
                    p_data_input = np.expand_dims(np.transpose((np.array(p_data_patch, np.float32)) / 255, (2, 0, 1)), 0)
                    param = {'input': p_data_input}
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
                if y_e == height:
                    roi_r_e = y_e
                else:
                    roi_r_e = round(y_e - process_size[0] * overlap / 2)

                if x_s == 0:
                    roi_c_s = 0
                else:
                    roi_c_s = round(x_s + process_size[0] * overlap / 2)
                if x_e == width:
                    roi_c_e = x_e
                else:
                    roi_c_e = round(x_e - process_size[0] * overlap / 2)
                modify_pr2 = modify_pr[roi_r_s - y_s:roi_r_e - y_s, roi_c_s - x_s:roi_c_e - x_s]
                predict_data[roi_r_s:roi_r_e, roi_c_s:roi_c_e] = modify_pr2
            pbar2.update(1)
        pbar1.update(1)

        for category in desired_classes:
            binary_mask = cv2.inRange(predict_data, category, category)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            binary_mask = cv2.dilate(binary_mask, kernel, iterations=2)
            binary_mask = cv2.erode(binary_mask, kernel, iterations=2)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if len(contour) >= 4:
                    contour = contour.squeeze()
                    contour[:, 0] = (contour[:, 0] + b_x_s) * x_cell_size + x_geo_start
                    contour[:, 1] = (contour[:, 1] + b_y_s) * y_cell_size + y_geo_start

                    geometry = {
                        "type": "Polygon",
                        "coordinates": [contour]
                    }
                    print("====================>")
                    print(CLASSES)
                    feature = {
                        "type": "Feature",
                        "properties": {"category": CLASSES[category], "category_value": category},
                        "geometry": geometry
                    }
                    all_features.append(feature)
    geojson = {
        "type": "FeatureCollection",
        "features": all_features
    }
    with open(output_path, 'w') as f:
        json_txt = gpd.GeoDataFrame.from_features(geojson).to_json()
        json_obj = json.loads(json_txt)
        json_obj['crs'] = {"type": "name", "properties": {"name": osr.SpatialReference(proj).ExportToProj4()}}
        f.write(json.dumps(json_obj))


if __name__ == '__main__':
    model_path = 'onnx_predict/models.onnx'
    # input_path = 'E:/ZEV/dockerruntime/geoserver/data/predict/DQ-4_017-DOM_345.tif'
    input_path = 'onnx_predict/predict.vrt'
    output_path = 'out/test.geojson'
    desired_classes = [1, 2, 4]
    region = [120.286043, 30.718273, 120.292024, 30.728884]
    # region = [120.290043, 30.728873, 120.292024, 30.728884]
    # region = [120.266192601, 30.706814747, 120.292356785, 30.729307369]

    # if not os.path.exists(os.path.dirname(output_path)):
    #     os.makedirs(os.path.dirname(output_path))
    # predict(model_path, input_path, output_path, desired_classes, region)

    parser = argparse.ArgumentParser()
    # parser.add_argument('-model_path', help='onnx模型路径')
    # parser.add_argument('-input_path', help='输入路径，vrt或tif')
    # parser.add_argument('-output_path', help='输出路径，geojson')
    # parser.add_argument('-class_name', type=str, nargs='+', default=[1,2], help='需要预测的类别')
    parser.add_argument('-class_name', type=str, help='预测的类别')
    parser.add_argument('-classes', type=str, help='需要预测的类别')
    parser.add_argument('-region', help='预测范围，经纬度坐标')

    # CLASSES = {
    # 0: 'background',
    # 1: 'building',
    # 2: 'glass',
    # 3: 'tree',
    # 4: 'water',
    # 5: 'tea'
    # }

    
    args = parser.parse_args()
    # model_path = args.model_path
    # input_path = args.input_path
    # output_path = args.output_path
    classes = eval(args.classes)
    # class_name = args.class_name
    CLASSES = ast.literal_eval(args.class_name)
    # class_name = {i: str(class_name[i]) for i in range(len(class_name))}
    # print(class_name)
    # CLASSES = class_name
    print(CLASSES)
    print(CLASSES[1])
    print(CLASSES[2])
    # print(CLASSES[str(classes[0])])
    # classes = [int(i) for i in classes]
    # region = args.region
    # if not os.path.exists(os.path.dirname(output_path)):
    #     os.makedirs(os.path.dirname(output_path))
    predict(model_path, input_path, output_path, classes, region)