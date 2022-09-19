import ast
import numpy as np
import torch
import torch.utils.data as data


class TemporalChunkCrop:
    def __init__(self, size: int = 4):
        self.S = size

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        if len(features) <= self.S:
            return features
        num_el = len(features) // self.S
        idxs = torch.tensor([self.S * i for i in range(num_el)])
        return idxs


class RandomTemporalChunkCrop:
    def __init__(self, size: int = 4):
        self.S = size

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        if len(features) <= self.S * 2:  # if we sample only one feature, everything breaks
            return features
        num_el = len(features) // self.S
        idxs = torch.randint(low=0, high=self.S, size=(num_el,))
        idxs = torch.tensor([j + self.S * i for i, j in enumerate(idxs.tolist())])
        return torch.index_select(features, 0, idxs)


class XDViolence(data.Dataset):
    def __init__(self, data_path: str, transform=None, test_mode: bool = False,
                 max_seqlen: int = 600, sample_num: int = 1, sample_window: int = 3, sample_strat: str = 'new'):

        self.test_mode = test_mode
        self.max_seqlen = max_seqlen
        self.sample_num = sample_num
        self.sample_strat = sample_strat
        self.sample_window = sample_window
        if test_mode:
            self.sampling = TemporalChunkCrop(size=sample_window)
            self.rgb_list_file = f"{data_path}/rgb_test_list.txt"
            self.flow_list_file = f"{data_path}/flow_test_list.txt"
            with open(f"{data_path}/segment_labels.txt", 'r') as f:
                string = f.read()
                self.segment_labels = ast.literal_eval(string)
            with open(f"{data_path}/annotations.txt", 'r') as f:
                string = f.read().replace('.mp4', '').splitlines()
                self.annotations = {s.split(' ')[0]: s.split(' ')[1:] for s in string}
        else:
            self.sampling = RandomTemporalChunkCrop(size=sample_window)
            self.rgb_list_file = f"{data_path}/rgb_list.txt"
            self.flow_list_file = f"{data_path}/flow_list.txt"

        self.transform = transform
        self.normal_flag = '_label_A'
        with open(self.rgb_list_file, 'r') as f:
            self.list = f.read().splitlines()

        self.list = [f"{data_path}/{x}" for x in self.list]

        self.features_len = 1024
        self.num_class = 1

    def __getitem__(self, index):
        if self.normal_flag in self.list[index]:
            video_label = torch.tensor(0.0)
        else:
            video_label = torch.tensor(1.0)

        features = torch.tensor(np.load(self.list[index]), dtype=torch.float32)
        vid_name = self.list[index].split('/')[-1].split('.npy')[0]

        if self.transform is not None:
            features = self.transform(features)

        if self.test_mode:
            frame_labels = torch.zeros(len(features) * 16, dtype=torch.float32)
            if self.sample_strat == 'new':
                idxs = self.sampling(features) if len(features) > self.sample_window * 2 else np.arange(len(features) - 1)
                features = features[idxs]
                frame_idxs = torch.cat([torch.tensor(np.arange(i * 16, i * 16 + 16)) for i in idxs])

            else:
                idxs = torch.range(0, len(features) - 1)

            if video_label == 0.0:
                segment_labels = torch.zeros(len(features), dtype=torch.float32)
            else:
                segment_labels = torch.tensor(self.segment_labels[vid_name[:-3]], dtype=torch.float32)[idxs]
                for start, end in zip(self.annotations[vid_name[:-3]][::2], self.annotations[vid_name[:-3]][1::2]):
                    frame_labels[int(start):int(end)] = 1.0

            return vid_name, features, video_label, segment_labels, frame_labels[frame_idxs]

        else:

            features = torch.cat([self.sampling(features) for _ in range(self.sample_num)])
            return vid_name, features, video_label, [len(features) // self.sample_num] * self.sample_num

    def __len__(self):
        return len(self.list)


if __name__ == '__main__':
    pass
