import os
import os.path as osp
from glob import glob
from torch.utils.data import Dataset
import numpy as np
import torch
from functools import partial
import torch.nn.functional as F


class CTDataset(Dataset):
    def __init__(self, dataset, mode, test_id=9, dose=5, context=True):
        print("dose",dose)
        self.mode = mode
        self.context = context
        print("dataset", dataset)

        if dataset in ['mayo_2016_sim', 'mayo_2016']:
            if dataset == 'mayo_2016_sim':
                data_root = './data_preprocess/gen_data/mayo_2016_sim_npy'
            elif dataset == 'mayo_2016':
                print("mayo_2016  mayo_2016_npy")
                data_root = './data_preprocess/gen_data/mayo_2016_npy'
                
            patient_ids = [67, 96, 109, 143, 192, 286, 291, 310, 333, 506]
            if mode == 'train':
                patient_ids.pop(test_id) #remove test_id
                # patient_ids = patient_ids[:test_id]
            # print("train patient_ids:", patient_ids)
            elif mode == 'test':
                patient_ids = patient_ids[test_id:test_id + 1]
                # patient_ids = patient_ids[test_id:]

            patient_lists = []
            for ind, id in enumerate(patient_ids):
                patient_list = sorted(glob(osp.join(data_root, ('L{:03d}_target_'.format(id) + '*_img.npy'))))
                patient_lists = patient_lists + patient_list[1:len(patient_list) - 1]
            base_target = patient_lists

            patient_lists = []
            for ind, id in enumerate(patient_ids):
                patient_list = sorted(glob(osp.join(data_root, ('L{:03d}_{}_'.format(id, dose) + '*_img.npy'))))  #zhiding dose
                if context:
                    cat_patient_list = []
                    for i in range(1, len(patient_list) - 1):
                        patient_path = ''
                        for j in range(-1, 2):
                            patient_path = patient_path + '~' + patient_list[i + j]
                        cat_patient_list.append(patient_path)
                    patient_lists = patient_lists + cat_patient_list
                else:
                    patient_list = patient_list[1:len(patient_list) - 1]
                    patient_lists = patient_lists + patient_list
            base_input = patient_lists

        elif dataset == 'Bern':
            data_root = './data_preprocess/gen_data/Bern_npy'
            # patient_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19,
            #                20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
            #                38, 39, 40, 41, 42, 43, 44, 45, 48, 49, 50, 51, 54, 55,
            #                56, 57, 58, 59, 60, 61, 62, 64, 65, 67, 68, 69, 71, 72, 73,
            #                74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91,
            #                92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108,
            #                109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124,
            #                125, 126, 127, 128, 129, 130, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141,
            #                142, 143, 144, 145, 146, 147, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158,
            #                159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
            #                176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192,
            #                193, 194, 195, 196, 197, 198, 199, 200, 201, 202]
            patient_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                           27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
                           53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
                           81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105,
                           106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126,
                           127, 128, 129, 130, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148,
                           149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169,
                           170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190,
                           191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202]
            print("test_id", test_id)
            if mode == 'train':
                index = patient_ids.index(test_id)
                # Slice the list to remove elements from test_id onward
                patient_ids = patient_ids[:index]
                # patient_ids.pop(test_id)  # remove test_id
                print("train patient_ids ", patient_ids )
            elif mode == 'test':
                # patient_ids = patient_ids[test_id:test_id + 1]
                index = patient_ids.index(test_id)
                # Slice the list to keep elements from test_id onward
                patient_ids = patient_ids[index:]
                print("test patient_ids ", patient_ids)
            patient_lists = []
            # print("**********patient_ids***********", patient_ids)
            for ind, id in enumerate(patient_ids):
                # print("id", id)
                patient_list = sorted(glob(osp.join(data_root, ('{}_target_z_'.format(id) + '*_png.npy'))))
                # print("patient_list", patient_list)
                # with open('patient_list.txt', 'a') as f:  # 使用 'a' 模式追加内容
                #     for file_path in patient_list:
                #         f.write(file_path + '\n')
                # print("patient_list[1:len(patient_list) - 1]", len(patient_list[1:len(patient_list) - 1]))
                patient_lists = patient_lists + patient_list[1:len(patient_list) - 1]
            # print("patient_lists", len(patient_lists))
            base_target = patient_lists

            patient_lists = []
            for ind, id in enumerate(patient_ids):
                patient_list = sorted(
                    glob(osp.join(data_root, ('{}_{}_z_'.format(id, dose) + '*_png.npy'))))  # zhiding dose
                if context:
                    cat_patient_list = []
                    for i in range(1, len(patient_list) - 1):
                        patient_path = ''
                        for j in range(-1, 2):
                            patient_path = patient_path + '~' + patient_list[i + j]
                            # print("####patient_path", patient_path)
                        cat_patient_list.append(patient_path)
                    # print("len(cat_patient_list)",len(cat_patient_list))#len(cat_patient_list) 118
                    patient_lists = patient_lists + cat_patient_list
                else:
                    patient_list = patient_list[1:len(patient_list) - 1]
                    patient_lists = patient_lists + patient_list
            base_input = patient_lists

        elif dataset == 'UI':
            data_root = './data_preprocess/gen_data/UI_npy'
            # patient_ids = [0, 2, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16,
            #                18, 20, 21, 24, 27, 29, 31, 33, 34, 35, 38, 39,
            #                 41, 45, 48, 51, 58, 59, 60, 64, 65, 68, 70, 78,
            #                 79, 81, 83, 88, 89, 93, 94, 99, 100, 104, 107,
            #                 109, 110, 111, 112, 116, 120, 121, 124, 125, 127,
            #                 128, 135, 136, 137, 142, 143, 147, 148, 152, 154,
            #                 155, 156, 157, 164, 166, 171, 175, 176, 177, 179,
            #                 180, 182, 184, 185, 186, 187, 190, 191, 192, 195,
            #                 196, 197, 199, 202, 204, 205, 208, 209, 212, 213,
            #                 214, 216, 217, 222, 224, 226, 228, 232, 234, 235,
            #                 240, 242, 245, 250, 252, 253, 259, 262, 266, 271,
            #                 272, 274, 276, 278, 279, 285, 287, 290, 295, 298,
            #                 303, 307, 308, 311, 312, 313, 314, 318]
            patient_ids = [0, 1, 2, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 29,
                           31, 33, 34, 35, 37, 38, 39, 40, 41, 43, 44, 45, 47, 48, 50, 51, 54, 55, 56, 57, 58, 59, 60,
                           61, 63, 64,
                           65, 68, 70, 72, 75, 76, 78, 79, 80, 81, 82, 83, 85, 88, 89, 90, 91, 92, 93, 94, 95, 97, 99,
                           100, 103,
                           104, 107, 108, 109, 110, 111, 112, 114, 116, 117, 120, 121, 123, 124, 125, 126, 127, 128,
                           129, 130, 132,
                           133, 134, 135, 136, 137, 139, 141, 142, 143, 144, 145, 146, 147, 148, 152, 153, 154, 155,
                           156, 157, 158,
                           159, 161, 163, 164, 166, 167, 169, 171, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182,
                           183, 184, 185,
                           186, 187, 188, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 202, 203, 204, 205, 206,
                           207, 208, 209,
                           211, 212, 213, 214, 215, 216, 217, 218, 219, 222, 223, 224, 226, 228, 231, 232, 233, 234,
                           235, 236, 238,
                           239, 240, 241, 242, 243, 245, 246, 249, 250, 251, 252, 253, 255, 256, 257, 259, 261, 262,
                           264, 265, 266,
                           267, 269, 270, 271, 272, 274, 275, 276, 277, 278, 279, 284, 285, 286, 287, 289, 290, 291,
                           292, 294, 295,
                           296, 297, 298, 299, 300, 301, 302, 303, 304, 306, 307, 308, 309, 310, 311, 312, 313, 314,
                           315, 317, 318, 319]
            print("test_id", test_id)
            if mode == 'train':
                index = patient_ids.index(test_id)
                # Slice the list to remove elements from test_id onward
                patient_ids = patient_ids[:index]
                # patient_ids.pop(test_id)  # remove test_id
                print("train patient_ids ", patient_ids )
            elif mode == 'test':
                # patient_ids = patient_ids[test_id:test_id + 1]
                index = patient_ids.index(test_id)
                # Slice the list to keep elements from test_id onward
                patient_ids = patient_ids[index:]
                print("test patient_ids ", patient_ids)
            patient_lists = []
            # print("**********patient_ids***********", patient_ids)
            for ind, id in enumerate(patient_ids):
                # print("id", id)
                patient_list = sorted(glob(osp.join(data_root, ('{}_target_'.format(id) + '*_png.npy'))))
                # print("patient_list", patient_list)
                # with open('patient_list.txt', 'a') as f:  # 使用 'a' 模式追加内容
                #     for file_path in patient_list:
                #         f.write(file_path + '\n')
                # print("patient_list[1:len(patient_list) - 1]", len(patient_list[1:len(patient_list) - 1]))
                patient_lists = patient_lists + patient_list[1:len(patient_list) - 1]
            print("patient_lists", len(patient_lists))
            base_target = patient_lists

            patient_lists = []
            for ind, id in enumerate(patient_ids):
                patient_list = sorted(
                    glob(osp.join(data_root, ('{}_{}_'.format(id, dose) + '*_png.npy'))))  # zhiding dose
                if context:
                    cat_patient_list = []
                    for i in range(1, len(patient_list) - 1):
                        patient_path = ''
                        for j in range(-1, 2):
                            patient_path = patient_path + '~' + patient_list[i + j]
                        cat_patient_list.append(patient_path)
                    patient_lists = patient_lists + cat_patient_list
                else:
                    patient_list = patient_list[1:len(patient_list) - 1]
                    patient_lists = patient_lists + patient_list
            base_input = patient_lists

        # elif dataset in ['png_pet']:
        #     print("run png_pet")
        #     data_root = './data_preprocess/gen_data/png_pet_npy'
        #
        #     patient_ids = [188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202]
        #     print("test_id", test_id)
        #     if mode == 'train':
        #         print("run train")
        #         # Ensure test_ids are sorted in reverse to avoid index shifting issues
        #         for test_id in sorted(test_ids, reverse=True):
        #             if 0 <= test_id < len(patient_ids):
        #                 patient_ids.pop(test_id)
        #             else:
        #                 print(f"Index {test_id} is out of range. Skipping.")
        #         print("Updated patient_ids:", patient_ids)
        #     elif mode == 'test':
        #         print("run test")
        #         selected_patient_ids = []
        #         for test_id in test_ids:
        #             # 确保 test_id 是有效的索引
        #             if 0 <= test_id < len(patient_ids):
        #                 selected_patient_ids.append(patient_ids[test_id])
        #             else:
        #                 print(f"Index {test_id} is out of range. Skipping.")
        #         # 将 selected_patient_ids 赋值给 patient_ids
        #         patient_ids = selected_patient_ids
        #         print("Selected patient_ids:", patient_ids)
        #     patient_lists = []
        #     for ind, id in enumerate(patient_ids):
        #         # print("#############")
        #         # patient_list = sorted(glob(osp.join(data_root, ('L{:03d}_target_'.format(id) + '*_img.npy'))))
        #         patient_list = sorted(glob(osp.join(data_root, ('{:03d}_target_'.format(id) + '*_png.npy'))))
        #         # print("patient_list ", patient_list)
        #         patient_lists = patient_lists + patient_list[1:len(patient_list) - 1]
        #     base_target = patient_lists
        #
        #     patient_lists = []
        #     for ind, id in enumerate(patient_ids):
        #         patient_list = sorted(
        #             # glob(osp.join(data_root, ('L{:03d}_{}_'.format(id, dose) + '*_img.npy'))))  # zhiding dose
        #             glob(osp.join(data_root, ('{:03d}_{}_'.format(id, dose) + '*_png.npy'))))  # zhiding dose
        #         if context:
        #             cat_patient_list = []
        #             for i in range(1, len(patient_list) - 1):
        #                 patient_path = ''
        #                 for j in range(-1, 2):
        #                     patient_path = patient_path + '~' + patient_list[i + j]
        #                 cat_patient_list.append(patient_path)
        #             patient_lists = patient_lists + cat_patient_list
        #         else:
        #             patient_list = patient_list[1:len(patient_list) - 1]
        #             patient_lists = patient_lists + patient_list
        #     base_input = patient_lists

   
    

        self.input = base_input
        self.target = base_target
        # print("self.input", self.input)
        # print("self.target", self.target)
        print("base_input", len(self.input))
        print("base_target", len(self.target))


    def __getitem__(self, index):
        input, target = self.input[index], self.target[index]
        # print("input", input)
        # print("target", target)
        if self.context:
            input = input.split('~')
            inputs = []
            for i in range(1, len(input)):
                inputs.append(np.load(input[i])[np.newaxis, ...].astype(np.float32))
            input = np.concatenate(inputs, axis=0)  #(3, 512, 512)
        else:
            input = np.load(input)[np.newaxis, ...].astype(np.float32) #(1, 512, 512)
        target = np.load(target)[np.newaxis,...].astype(np.float32) #(1, 512, 512)
        input = input / 255.0  #pet归一化到 [0, 1]
        target = target / 255.0  #归一化到 [0, 1]
        # input = self.normalize_(input)  #CT
        # target = self.normalize_(target)   #CT

        return input, target

    def __len__(self):
        return len(self.target)

    # def normalize_(self, img, MIN_B=-1024, MAX_B=3072):
    #     img = img - 1024
    #     img[img < MIN_B] = MIN_B
    #     img[img > MAX_B] = MAX_B
    #     img = (img - MIN_B) / (MAX_B - MIN_B)
    #     return img


dataset_dict = {
    'train': partial(CTDataset, dataset='mayo_2016_sim', mode='train', test_id=9, dose=25, context=True), #'mayo_2016_sim'
    'mayo_2016_sim': partial(CTDataset, dataset='mayo_2016_sim', mode='test', test_id=9, dose=25, context=True),
    'mayo_2016': partial(CTDataset, dataset='mayo_2016', mode='test', test_id=9, dose=25, context=True),
    'png_pet': partial(CTDataset, dataset='png_pet', mode='test', test_id=9, dose=25, context=True),
    'Bern': partial(CTDataset, dataset='Bern', mode='test', test_id=183, dose=25, context=True),
    'UI': partial(CTDataset, dataset='UI', mode='test', test_id=9, dose=25, context=True),
    'mayo_2020': partial(CTDataset, dataset='mayo_2020', mode='test', test_id=None, dose=None, context=True),
    'piglet': partial(CTDataset, dataset='piglet', mode='test', test_id=None, dose=None, context=True),
    'phantom': partial(CTDataset, dataset='phantom', mode='test', test_id=None, dose=108, context=True),
}
