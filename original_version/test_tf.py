# file_path = "test_filename.txt"
# import scipy.misc as misc
# import scipy.io as scio
# import numpy as np
# import os
# import cv2
#
#
# f = open(file_path)
# gt_folder = "E:\ISBI_code\Source_Data\\650\\650mask"
# all_files = f.readlines()
#
#
#
#
# def preprocessed_mat(image_path):
#     data = scio.loadmat(image_path)
#     # print(type(data))
#     # print(data['maskFull'])
#     # print(data.keys())
#     resized_data = data['maskFull']
#     # resized_data = misc.imresize(original_data, (1024, 1024))
#     resized_data = resized_data / (np.max(resized_data) - np.min(resized_data)) * 255.
#     return resized_data
#
# for test_file in all_files[:8]:
#     print(test_file)
#     test_file_path = os.path.join(gt_folder, test_file.split('.')[0]+'.mat')
#     gt = preprocessed_mat(test_file_path)
#     cv2.imwrite(os.path.join("E:\ISBI_code\original_version\data", test_file.split('.')[0]+'.jpg'), gt)
import csv
label_path = "/Users/imed-05/Downloads/ISBI_code/Source_Data/labels.xlsx"
# with open(label_path) as csvfile:
#     reader = csv.DictReader(csvfile)
#     for row in reader:
#         print(row)
from openpyxl import load_workbook
wb = load_workbook(label_path)

sheets = wb.get_sheet_names()
sheet_first = sheets[0]
# print(sheets)
print(sheets[0])
ws = wb.get_sheet_by_name(sheet_first)
rows = ws.rows


#
# for file_name in all_files_name:
#     fp.write(file_name+"\n")
# fp.close()
for i, row in enumerate(rows):
    if i>0:
        line = [col.value for col in row]

        fp.write(line[1]+'    '+str(line[-1])+"\n")
fp.close()
