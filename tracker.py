from __future__ import print_function, division

import os
import numpy as np

import torch
from torch.utils import data
from torchvision import datasets, transforms
# from torchvision.transforms import InterpolationMode

from path_use import *

"""
修改时间：2022/5/2
修改内容：
1、self.get_cam_id_and_time函数中增加了位置信息
    self.gallery_cam, self.pic_num, self.place_num = self.get_cam_id_and_time(gallery_path)
2、self.get_label函数中修改了从query中获取图片信息
    label = file_name.split('_')[-1].split('.')[0]
3、self.get_cam_id_and_time增加了从图片名称中获取地点的语句
place = filename.split('_')[0]
4、self.compute_distance函数中加入了 self.place_num
temp = [self.query_label[query_num], self.gallery_cam[i], self.pic_num[i], self.place_num[i]]
5、self.compute_distance函数中增加了时间戳文件名。
timestamp = get_timestamp()
self.my_npy_name = os.path.join(self.params.data_output, 'npy', timestamp + 'rm.npy')
"""


class Tracker:
    """

    """

    def __init__(self, ie, params, device, data_dir):
        """

        :param ie: IE推理模型
        :param params: 输入的参数
        :param device: 设备，如"CPU"
        :param data_dir: 数据所在路径
        """
        self.ie = ie
        self.params = params
        # self.query_dir = query_dir
        # self.gallery_dir = gallery_dir
        self.model_xml = self.params.m_reid_xml
        self.model_bin = self.params.m_reid_bin
        self.device = device
        self.data_dir = data_dir

        self.input_blob = None
        self.out_blob = None
        self.my_npy_name = None

    def load_network(self):
        """

        导入网络
        :return: 返回网络和
        """
        net = self.ie.read_network(model=self.model_xml, weights=self.model_bin)
        self.input_blob = next(iter(net.input_info))
        self.out_blob = next(iter(net.outputs))
        exec_net = self.ie.load_network(network=net, num_requests=1, device_name=self.device)
        return net, exec_net

    def fliplr(self, img):
        """flip horizontal"""
        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
        img_flip = img.index_select(3, inv_idx)
        return img_flip

    def get_feature_map(self):
        """
        获得特征图片
        :return: 特征图片
        """
        h, w = 224, 224
        print("[tracker.py] Deploying")
        data_transforms = transforms.Compose([transforms.Resize((h, w)),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        data_dir = self.data_dir
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in
                          ['gallery', 'query']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                                      shuffle=False, num_workers=0) for x in ['gallery', 'query']}

        gallery_path = image_datasets['gallery'].imgs
        # print(f'gallery_path:{gallery_path}')
        query_path = image_datasets['query'].imgs

        self.gallery_cam, self.pic_num, self.place_num = self.get_cam_id_and_time(gallery_path)
        self.query_label = self.get_label(query_path)

        gallery_feature = self.extract_feature(dataloaders['gallery'])
        query_feature = self.extract_feature(dataloaders['query'])

        feature_map = {'gallery_f': gallery_feature.numpy(), 'gallery_cam': self.gallery_cam, 'pic_num': self.pic_num,
                       'query_f': query_feature.numpy(), 'query_label': self.query_label}
        # print(feature_map['query_label'],feature_map['query_f'])
        print("[tracker.py]  Finished deploying.")
        return feature_map

    def extract_feature(self, dataloaders):
        """

        :param dataloaders: torch的 Dataloaders
        :return:
        """
        net, exec_net = self.load_network()
        for iters, data in enumerate(dataloaders):
            img, label = data
            n, c, h, w = img.size()
            ff = torch.FloatTensor(n, 512).zero_()
            for i in range(2):
                if i == 1:
                    img = self.fliplr(img)
                input_img = img

                # print('Now inference:',img)
                outputs = self.inference(exec_net, input_img)
                # print('out',outputs)
                ff += outputs
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))
            # print('ff',ff)
            if iters == 0:
                features = torch.FloatTensor(len(dataloaders.dataset), ff.shape[1])
            start = iters
            end = min((iters + 1), len(dataloaders.dataset))
            features[start:end, :] = ff

        return features

    def inference(self, exec_net, input_img):
        result = exec_net.infer(inputs={self.input_blob: input_img})
        return result['output_1']

    def get_label(self, q_path):
        try:
            label_list = []
            for path, v in q_path:
                file_name = os.path.basename(path)
                label = file_name.split('_')[-1].split('.')[0]
                # print(label)
                label_list.append(int(label[0]))
            return label_list
        except:
            print('Please rename picture:{}, e.g. 0.jpg'.format(file_name))

    def get_cam_id_and_time(self, g_path):
        place_id = []
        camera_id = []
        pic_info = []
        for path, v in g_path:
            filename = os.path.basename(path)
            # print(f'filename:{filename}, path:{path}')
            place = filename.split('_')[0]
            camera = filename.split('_')[-3]
            time_num = int(filename.split('_')[-2])
            person_num = int(filename.split('_')[-1][0])
            if camera == '-1':
                continue
            place_id.append(place)
            camera_id.append(int(camera[0]))
            pic_info.append([time_num, person_num])

        return camera_id, pic_info, place_id

    def sort_img(self, qf, gf):
        query = qf.view(-1, 1)
        # print(query.shape)
        score = torch.mm(gf, query)
        score = score.squeeze(1)
        score = score.numpy()
        # print('score',score)
        # predict index
        index = np.argsort(score)  # if no "-", then from small to large
        index = index[::-1]
        return index

    def compute_distance(self, feature_map):
        print("[tracker.py] Computing……")
        raw_message = []
        gallery_f = torch.FloatTensor(feature_map['gallery_f'])
        for query_num in range(len(self.query_label)):
            query_f = torch.FloatTensor(feature_map['query_f'][query_num])
            index = self.sort_img(query_f, gallery_f)
            choose = index[:self.params.top_k]
            # print(index)
            # print(self.gallery_cam)
            # print(self.pic_num)
            for i in choose:
                temp = [self.query_label[query_num], self.gallery_cam[i], self.pic_num[i], self.place_num[i]]
                raw_message.append(temp)
        if self.params.is_save:
            rm = np.array(raw_message, dtype=object)

            timestamp = get_timestamp()
            self.my_npy_name = os.path.join(self.params.data_output, 'npy', timestamp + 'rm.npy')
            # self.my_npy_name = self.params.output_npy_path + timestamp + 'rm.npy'

            # my_npy_name = self.params.output_npy_path + 'rm.npy'
            # name_count = 0
            # while os.path.isfile(my_npy_name):
            #     my_npy_name = self.params.output_npy_path + 'rm_' + str(name_count) + '.npy'
            #     name_count += 1

            np.save(self.my_npy_name, rm)
            print(f"[tracker.py] {self.my_npy_name} save successfully.")
        print("[tracker.py] Finished Computing.")
        return raw_message
