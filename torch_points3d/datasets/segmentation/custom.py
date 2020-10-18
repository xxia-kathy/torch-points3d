import os
import os.path as osp
from itertools import repeat, product
import numpy as np
import h5py
import torch
import random
import glob
from plyfile import PlyData, PlyElement
from torch_geometric.data import InMemoryDataset, Data, extract_zip, Dataset
from torch_geometric.data.dataset import files_exist
from torch_geometric.data import DataLoader
from torch_geometric.datasets import S3DIS as S3DIS1x1
import torch_geometric.transforms as T
import logging
from sklearn.neighbors import NearestNeighbors, KDTree
from tqdm.auto import tqdm as tq
import csv
import pandas as pd
import pickle
import gdown
import shutil
from plyfile import PlyData, PlyElement

from torch_points3d.datasets.samplers import BalancedRandomSampler
import torch_points3d.core.data_transform as cT
from torch_points3d.datasets.base_dataset import BaseDataset

DIR = os.path.dirname(os.path.realpath(__file__))
log = logging.getLogger(__name__)

S3DIS_NUM_CLASSES = 2

INV_OBJECT_LABEL = {
    0: "Background",
    1: "Target",
}

OBJECT_COLOR = np.asarray(
    [
        [233, 229, 107],  # 'Background' .-> .yellow
        [95, 156, 196],  # 'Color' .-> . blue
    ]
)

OBJECT_LABEL = {name: i for i, name in INV_OBJECT_LABEL.items()}


################################### UTILS #######################################

def read_mesh_vertices_rgb(filename):
    """ read XYZ RGB for each vertex.
    Note: RGB values are in 0-255
    """
    assert os.path.isfile(filename)
    with open(filename, "rb") as f:
        plydata = PlyData.read(f)
        num_verts = plydata["vertex"].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:, 0] = plydata["vertex"].data["x"]
        vertices[:, 1] = plydata["vertex"].data["y"]
        vertices[:, 2] = plydata["vertex"].data["z"]
        vertices[:, 3] = plydata["vertex"].data["red"]
        vertices[:, 4] = plydata["vertex"].data["green"]
        vertices[:, 5] = plydata["vertex"].data["blue"]
    return vertices

class Custom(Dataset):
    def __init__(self, root, split, transform=None, pre_transform=None):
        super(CustomDataset, self).__init__(root, transform, pre_transform)
        
        if split == "train":
            path = self.processed_paths[0]
        elif split == "val":
            path = self.processed_paths[1]
        elif split == "test":
            path = self.processed_paths[2]
        else:
            raise ValueError((f"Split {split} found, but expected either " "train, val, or test"))

        self.data, self.slices = torch.load(path)

    @property
    def num_classes(self):
        return 2

# def object_name_to_label(object_class):
#     """convert from object name in S3DIS to an int"""
#     object_label = OBJECT_LABEL.get(object_class, OBJECT_LABEL["clutter"])
#     return object_label


# def read_data_format(train_file, label_out=True, verbose=False, debug=False):
#     """extract data from a room folder"""

#     raw_path = osp.join(train_file, "*.ply")
#     room_ver = pd.read_csv(raw_path, sep=" ", header=None).values
#     xyz = np.ascontiguousarray(room_ver[:, 0:3], dtype="float32")
#     try:
#         rgb = np.ascontiguousarray(room_ver[:, 3:6], dtype="uint8")
#     except ValueError:
#         rgb = np.zeros((room_ver.shape[0], 3), dtype="uint8")
#         log.warning("WARN - corrupted rgb data for file %s" % raw_path)
#     if not label_out:
#         return xyz, rgb
#     n_ver = room_ver.shape[0]
#     del room_ver
#     semantic_labels = np.zeros((n_ver,), dtype="int64")
#     labels = pd.read_csv(osp.join(train_file, "Annotations/*.txt"), sep=" ", header=None).values
#     for i, label in enumerate(labels):
#         pixel_label = object_name_to_label(label)
#         semantic_labels[i] = pixel_label
#     return (
#         torch.from_numpy(xyz),
#         torch.from_numpy(rgb),
#         torch.from_numpy(semantic_labels),        )
#     def __init__(self, root, split="trainval", transform=None, process_workers=1, pre_transform=None):
#         assert self.REMAPPING_MAP[0] == IGNORE_LABEL  # Make sure we have the same convention for unlabelled data
#         self.use_multiprocessing = process_workers > 1
#         self.process_workers = process_workers

#         super().__init__(root, transform=transform, pre_transform=pre_transform)
#         if split == "train":
#             self._scans = glob(os.path.join(self.processed_paths[0], "*.pt"))
#         elif split == "val":
#             self._scans = glob(os.path.join(self.processed_paths[1], "*.pt"))
#         elif split == "test":
#             self._scans = glob(os.path.join(self.processed_paths[2], "*.pt"))
#         elif split == "trainval":
#             self._scans = glob(os.path.join(self.processed_paths[0], "*.pt")) + glob(
#                 os.path.join(self.processed_paths[1], "*.pt")
#             )
#         else:
#             raise ValueError("Split %s not recognised" % split)

#     @property
#     def raw_file_names(self):
#         return [os.path.join("dataset", "sequences")]

#     @property
#     def processed_file_names(self):
#         return [s for s in self.AVAILABLE_SPLITS[:-1]]

#     def _load_paths(self, seqs):
#         scan_paths = []
#         label_path = []
#         for seq in seqs:
#             scan_paths.extend(
#                 sorted(glob(os.path.join(self.raw_paths[0], "{0:02d}".format(int(seq)), "velodyne", "*.bin")))
#             )
#             label_path.extend(
#                 sorted(glob(os.path.join(self.raw_paths[0], "{0:02d}".format(int(seq)), "labels", "*.label")))
#             )

#         if len(label_path) == 0:
#             label_path = [None for i in range(len(scan_paths))]
#         if len(label_path) > 0 and len(scan_paths) != len(label_path):
#             raise ValueError((f"number of scans {len(scan_paths)} not equal to number of labels {len(label_path)}"))

#         return scan_paths, label_path

#     @staticmethod
#     def read_raw(scan_file, label_file=None):
#         scan = np.fromfile(scan_file, dtype=np.float32).reshape(-1, 4)
#         data = Data(pos=torch.tensor(scan[:, :3]), x=torch.tensor(scan[:, 3]).reshape(-1, 1),)
#         if label_file:
#             label = np.fromfile(label_file, dtype=np.uint32).astype(np.int32)
#             assert scan.shape[0] == label.shape[0]
#             semantic_label = label & 0xFFFF
#             instance_label = label >> 16
#             data.y = torch.tensor(semantic_label).long()
#             data.instance_labels = torch.tensor(instance_label).long()
#         return data

#     @staticmethod
#     def process_one(scan_file, label_file, transform, out_file):
#         data = SemanticKitti.read_raw(scan_file, label_file)
#         if transform:
#             data = transform(data)
#         log.info("Processed file %s, nb points = %i", os.path.basename(out_file), data.pos.shape[0])
#         torch.save(data, out_file)

#     def get(self, idx):
#         data = torch.load(self._data[idx])
#         if data.y is not None:
#             data.y = self._remap_labels(data.y)
#         return data

#     def process(self):
#         for i, split in enumerate(self.AVAILABLE_SPLITS[:-1]):
#             if os.path.exists(self.processed_paths[i]):
#                 continue
#             os.makedirs(self.processed_paths[i])

#             seqs = self.SPLIT[split]
#             pc_paths, label_paths = self._load_paths(seqs)
#             pc_names = []
#             for pc in pc_paths:
#                 pc = os.path.splitext(pc)[0]
#                 seq, _, pc_id = pc.split(os.path.sep)[-3:]
#                 pc_names.append("{}_{}".format(seq, pc_id))

#             out_files = [os.path.join(self.processed_paths[i], "{}.pt".format(pc_name)) for pc_name in pc_names]
#             args = zip(pc_paths, label_paths, [self.pre_transform for i in range(len(pc_paths))], out_files)
#             if self.use_multiprocessing:
#                 with multiprocessing.Pool(processes=self.process_workers) as pool:
#                     pool.starmap(self.process_one, args)
#             else:
#                 for arg in args:
#                     self.process_one(*arg)

#     def len(self):
#         return len(self._data)

#     @property
#     def num_classes(self):
#         return 2


######################################################################


class CustomDataset(BaseDataset):
    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)
        self.train_dataset = Custom(
            self._data_path,
            split="train",
            transform=self.train_transform,
            pre_transform=self.pre_transform,
        )

        self.val_dataset = Custom(
            self._data_path,
            split="val",
            transform=self.val_transform,
            pre_transform=self.pre_transform,
        )

        self.test_dataset = Custom(
            self._data_path,
            split="test",
            transform=self.test_transform,
            pre_transform=self.pre_transform,
        )
    
    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker

        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        from torch_points3d.metrics.segmentation_tracker import SegmentationTracker

        return SegmentationTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)