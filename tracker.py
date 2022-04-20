from __future__ import print_function, division
import torch
import numpy as np
from torchvision import datasets, transforms
import os


class Tracker:
    def __init__(self, ie, params, device, data_dir):
        self.ie = ie
        self.params = params
        # self.query_dir = query_dir
        # self.gallery_dir = gallery_dir
        self.model_xml = self.params.m_reid_xml
        self.model_bin = self.params.m_reid_bin
        self.device = device
        self.data_dir = data_dir

    def load_network(self):
        net = self.ie.read_network(model=self.model_xml, weights=self.model_bin)
        self.input_blob = next(iter(net.input_info))
        self.out_blob = next(iter(net.outputs))
        exec_net = self.ie.load_network(network=net, num_requests=1, device_name=self.device)
        return net, exec_net

    def fliplr(self, img):
        """ flip horizontal """
        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
        img_flip = img.index_select(3, inv_idx)
        return img_flip

    def get_feature_map(self):
        """
        use openvino model to inference
        :return: feature map
        """
        h, w = 224, 224

        data_transforms = transforms.Compose([transforms.Resize((h, w), interpolation=3),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        data_dir = self.data_dir
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms) for x in
                          ['gallery', 'query']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                                      shuffle=False, num_workers=0) for x in ['gallery', 'query']}

        gallery_path = image_datasets['gallery'].imgs
        query_path = image_datasets['query'].imgs

        self.gallery_cam, self.pic_num = self.get_cam_id_and_time(gallery_path)
        self.query_label = self.get_label(query_path)

        gallery_feature = self.extract_feature(dataloaders['gallery'])
        query_feature = self.extract_feature(dataloaders['query'])

        feature_map = {'gallery_f': gallery_feature.numpy(), 'gallery_cam': self.gallery_cam, 'pic_num': self.pic_num,
                       'query_f': query_feature.numpy(), 'query_label': self.query_label}

        return feature_map

    def extract_feature(self, dataloaders):
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
                outputs = self.inference(net, exec_net, input_img)
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

    def inference(self, net, exec_net, input_img):
        result = exec_net.infer(inputs={self.input_blob: input_img})
        return result['output_1']

    def get_label(self, q_path):
        label_list = []
        for path, v in q_path:
            file_name = os.path.basename(path)
            label = file_name.split('_')[0]
            label_list.append(int(label[0]))
        return label_list

    def get_cam_id_and_time(self, g_path):
        camera_id = []
        pic_info = []
        for path, v in g_path:
            filename = os.path.basename(path)
            camera = filename.split('_')[0]
            time_num = int(filename.split('_')[1])
            person_num = int(filename.split('_')[2][0])
            if camera == '-1':
                continue
            camera_id.append(int(camera[0]))
            pic_info.append([time_num, person_num])

        return camera_id, pic_info

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
        print('Computing......')
        raw_message = []
        gallery_f = torch.FloatTensor(feature_map['gallery_f'])
        for query_num in range(len(self.query_label)):
            query_f = torch.FloatTensor(feature_map['query_f'][query_num])
            index = self.sort_img(query_f, gallery_f)
            choose = index[:self.params.top_k]
            for i in choose:
                temp = [self.query_label[query_num], self.gallery_cam[i], self.pic_num[i]]
                # pic_num:[frame_num,person_num]
                raw_message.append(temp)
        print('Finished computing.')
        return raw_message
