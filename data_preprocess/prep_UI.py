import os
import os.path as osp
import argparse
import numpy as np
from natsort import natsorted
from glob import glob
import pydicom
from PIL import Image
import matplotlib.pyplot as plt


def save_dataset(args):
    if not osp.exists(args.save_path):
        os.makedirs(args.save_path)
        print('Create path : {}'.format(args.save_path))

    # patient_ids = ['L067', 'L096', 'L109', 'L143', 'L192', 'L286', 'L291', 'L310', 'L333', 'L506']
    # patient_ids = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
    #  '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39',
    #  '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58',
    #  '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77',
    #  '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96',
    #  '97', '98', '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113',
    #  '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126', '127', '128', '129',
    #  '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140', '141', '142', '143', '144', '145',
    #  '146', '147', '148', '149', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', '160', '161',
    #  '162', '163', '164', '165', '166', '167', '168', '169', '170', '171', '172', '173', '174', '175', '176', '177',
    #  '178', '179', '180', '181', '182', '183', '184', '185', '186', '187', '188', '189', '190', '191', '192', '193',
    #  '194', '195', '196', '197', '198', '199', '200', '201', '202']

    patient_ids = [
    '0', '2', '4', '5', '6', '7', '11', '12', '13', '14', '15', '16',
    '18', '20', '21', '24', '27', '29', '31', '33', '34', '35', '38', '39',
    '41', '45', '48', '51', '58', '59', '60', '64', '65', '68', '70', '78',
    '79', '81', '83', '88', '89', '93', '94', '99', '100', '104', '107',
    '109', '110', '111', '112', '116', '120', '121', '124', '125', '127',
    '128', '135', '136', '137', '142', '143', '147', '148', '152', '154',
    '155', '156', '157', '164', '166', '171', '175', '176', '177', '179',
    '180', '182', '184', '185', '186', '187', '190', '191', '192', '195',
    '196', '197', '199', '202', '204', '205', '208', '209', '212', '213',
    '214', '216', '217', '222', '224', '226', '228', '232', '234', '235',
    '240', '242', '245', '250', '252', '253', '259', '262', '266', '271',
    '272', '274', '276', '278', '279', '285', '287', '290', '295', '298',
    '303', '307', '308', '311', '312', '313', '314', '318']

    io = 'target'
    # print("args.data_path", args.data_path)
    # patient_path1 = osp.join(args.data_path, 'full_1mm')
    patient_path1 = osp.join(args.data_path, 'UI_FPET_train')
    print("patient_path1", patient_path1)
    for p_ind, patient_id in enumerate(patient_ids):
        print(patient_id)
        if p_ind >= 0:
            # patient_path = osp.join(patient_path1, patient_id, 'full_1mm')
            patient_path = osp.join(patient_path1, patient_id)
            print("patient_path", patient_path)
            # data_paths = natsorted(glob(osp.join(patient_path, '*.IMA')))
            data_paths = natsorted(glob(osp.join(patient_path, 'nr*_z*.png')))
            print("data_paths", data_paths)
            for slice, data_path in enumerate(data_paths):
                # im = pydicom.dcmread(data_path)
                # f = np.array(im.pixel_array)
                im = Image.open(data_path)
                # im_resized = im.resize((256, 256))
                # f = np.array(im_resized)
                f = np.array(im)
                # f_name = '{}_{}_{:0>3d}_img.npy'.format(patient_id, io, slice)
                f_name = '{}_{}_{:0>3d}_png.npy'.format(patient_id, io, slice)
                # print("patient_id[2:]", patient_id[2:])
                # f_name = '{}_{}_{:0>3d}_png.npy'.format(patient_id[2:], io, slice)
                np.save(osp.join(args.save_path, f_name), f.astype(np.uint16))


    io = '25' # no clear
    # patient_path2 = osp.join(args.data_path, 'quarter_1mm')
    patient_path2 = osp.join(args.data_path, 'UI_LPET_train')
    for p_ind, patient_id in enumerate(patient_ids):
        print(patient_id)
        if p_ind >= 0:
            # patient_path = osp.join(patient_path2, patient_id, 'quarter_1mm')
            patient_path = osp.join(patient_path2, patient_id)
            # data_paths = natsorted(glob(osp.join(patient_path, '*.IMA')))
            data_paths = natsorted(glob(osp.join(patient_path, 'nr*_z*.png')))
            for slice, data_path in enumerate(data_paths):
                # im = pydicom.dcmread(data_path)
                # f = np.array(im.pixel_array)
                im = Image.open(data_path)
                width, height = im.size
                print(f"Width: {width}, Height: {height}")
                # 调整图像大小为 256x256 像素
                # im_resized = im.resize((256, 256))
                # f = np.array(im_resized)
                f = np.array(im)
                # f_name = '{}_{}_{:0>3d}_img.npy'.format(patient_id, io, slice)
                f_name = '{}_{}_{:0>3d}_png.npy'.format(patient_id, io, slice)
                # f_name = '{}_{}_{:0>3d}_png.npy'.format(patient_id[2:], io, slice)
                np.save(osp.join(args.save_path, f_name), f.astype(np.uint16))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/data/sunqi/code_python/CoreDiff-main/data_preprocess/UI_dataset/')   # data format: dicom
    parser.add_argument('--save_path', type=str, default='./gen_data/UI_npy/')
    args = parser.parse_args()

    save_dataset(args)
