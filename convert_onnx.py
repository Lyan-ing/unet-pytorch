import argparse
import json
import os.path

from loguru import logger
from unet import Unet

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_save_path', type=str, default='log_1',
                        help='onnx转换后的保存路径')
    parser.add_argument('--model_pth_path', type=str, default='log_1',
                        help='模型pth文件的读取路径，路径下的model_param.json需要存在')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    #-------------------------------------------------------------------------#
    #   如果想要修改对应种类的颜色，到__init__函数里修改self.colors即可
    #-------------------------------------------------------------------------#
    #----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'           表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'             表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'               表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'dir_predict'       表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    #   'export_onnx'       表示将模型导出为onnx，需要pytorch1.7.1以上。
    #   'predict_onnx'      表示利用导出的onnx模型进行预测，相关参数的修改在unet.py_346行左右处的Unet_ONNX
    #----------------------------------------------------------------------------------------------------------#
    #-------------------------------------------------------------------------#
    #   simplify            使用Simplify onnx
    #   onnx_save_path      指定了onnx的保存路径
    #-------------------------------------------------------------------------#
    args = args_parser()
    with open(os.path.join( args.model_pth_path, 'model_param.json'), 'r') as f:
        num_classes = len(json.load(f)["category"]) + 1
    simplify        = True
    unet = Unet(model_path = os.path.join(args.model_pth_path,'best_epoch_weights.pth'), num_classes= num_classes)
    unet.convert_to_onnx(simplify, os.path.join(args.onnx_save_path, 'model.onnx'))
    logger.info("模型转换完成")


