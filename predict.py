#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# PROGRAMMER: Lin Shih Chieh
# DATE CREATED: 2018-07-25
# 目的: 使用 predict.py 预测图像的花卉名称以及该名称的概率
#   基本用途： python predict.py input checkpoint
#   选项：
#     返回前 个类别： python predict.py input checkpoint ‐‐top_k 3
#     使用类别到真实名称的映射： python predict.py input checkpoint ‐‐category_names cat_to_name.json
#     使用 GPU 进行预测： python predict.py input checkpoint ‐‐gpu
#   Example call:
#    python predict.py input checkpoint ‐‐top_k 3 ‐‐category_names cat_to_name.json ‐‐gpu

# Import all functions
from predict_func import *

# Get command line arguments
args = get_input_args()

# Get category names of flowers from file
cat_names_dict = get_category_names(args.category_names)
print("花卉类别名称文件加载完成!")

# Load model from checkpoint
model = load_checkpoint(args.save_dir, args.checkpoint)
print("{} 网络模型加载完成!".format(args.checkpoint))

# Predict input image class
image_path = args.input
ps, ids = predict(image_path, model, args.gpu, args.top_k)

# Print predict result
ns = []
n = 0
topk_str = ''
cti = model.class_to_idx
for idx in ids:
    cat_name = (cat_names_dict[list(cti.keys())[list(cti.values()).index(int(idx))]]).capitalize()
    ns.append(cat_name)
    topk_str += "<{}> {}({:.2f}%) ".format(n+1, cat_name, ps[n]*100)
    n += 1

print("*"*80)
print("The Flower is : {} ({:.2f}%)".format(ns[0],ps[0]*100))
if args.top_k:
    print("Top K Classes : {}".format(topk_str))