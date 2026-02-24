import torch.utils.data as util_data
from torchvision import transforms
import torch
from PIL import Image
import numpy as np
from tqdm  import  tqdm
import lmdb
import pickle
import six
import os.path as osp

def one_hot(arr, idx):
    arr[idx] = 1
    return arr.astype(int)

class ImageList(object):
    def __init__(self, data_path, image_list, transform, num_class=1000):
        if len(image_list[0].split()) > 2:
            self.imgs = [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in
                         image_list]
        else:
            self.imgs = [(data_path + val.split()[0],  one_hot(np.zeros([num_class]),
                                                                   int(val.split()[1]))) for val in image_list]
        # imgs = "path" "label-one hot"
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.imgs)


def loads_data(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pickle.loads(buf)

class ImageFolderLMDB(util_data.Dataset):
    def __init__(self, db_path, transform=None, target_transform=None, num_class=1000):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)

        with self.env.begin(write=False) as txn:
            self.length = loads_data(txn.get(b'__len__'))
            self.keys = loads_data(txn.get(b'__keys__'))

        self.transform = transform
        self.target_transform = target_transform
        self.num_classes = num_class

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = loads_data(byteflow)

        # load img
        imgbuf = unpacked[0]
        img = imgbuf.convert('RGB')
        # load label
        target = unpacked[1]
        target = one_hot(np.zeros([self.num_classes]), int(target))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        # return img, target
        return img, target, index   # index is shuffled, can't be used directly

    def __len__(self):
       return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH

def CalcHammingDist_CUDA(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - torch.matmul(B1, B2.t()))
    return distH

L2Dist = torch.nn.PairwiseDistance(p=2)

def CalcL2Dist_CUDA(d1, d2):
    d1 = d1.unsqueeze(1)
    d2 = d2.unsqueeze(0)
    diff = d1 - d2
    distances = torch.sqrt(torch.sum(diff ** 2, dim=-1))
    return distances


def count_ones(xor_res):
    # the input is unsigned integer type
    xor_res = np.bitwise_and(xor_res, 0x55555555) + np.bitwise_and(np.right_shift(xor_res, 1), 0x55555555)
    xor_res = np.bitwise_and(xor_res, 0x33333333) + np.bitwise_and(np.right_shift(xor_res, 2), 0x33333333)
    xor_res = np.bitwise_and(xor_res, 0x0f0f0f0f) + np.bitwise_and(np.right_shift(xor_res, 4), 0x0f0f0f0f)
    xor_res = np.bitwise_and(xor_res, 0x00ff00ff) + np.bitwise_and(np.right_shift(xor_res, 8), 0x00ff00ff)
    xor_res = np.bitwise_and(xor_res, 0x0000ffff) + np.bitwise_and(np.right_shift(xor_res, 16), 0x0000ffff)

    return xor_res

def calculate_distance(qB, rB, cat='hamming'):
    if cat == 'hamming':  # qB and rB are uint32 type, 1*(bit_num//32), N*(bit_num//32)
        res = np.bitwise_xor(qB, rB)
        res = count_ones(res)
        distH = res.sum(1)
    elif cat == 'euclidean':
        dif = qB - rB
        distH = np.sqrt(np.sum(np.square(dif), 1))
    else:
        raise(RuntimeError('Unsupported distance category!'))

    return distH


def compress_binary_to_uint(hash_codes):
    print("compressing the bit string into integers...")
    if np.min(hash_codes) < 0:
        hash_codes[hash_codes <= 0] = 0
        # hash_codes[hash_codes > 0] = 1
    hash_codes = hash_codes.astype(np.uint32)
    num = hash_codes.shape[0]
    bits = hash_codes.shape[1]
    int_bits = 32
    assert bits % int_bits == 0, 'the bit num cannot be divided by that of an integer'
    comp_hash_codes = np.zeros((num, bits // int_bits), dtype=np.uint32)  # set int32 or uint32
    for m in range(bits):
        tmp = int(0)
        tmp = np.bitwise_or(tmp, hash_codes[:, m])
        tmp = np.left_shift(tmp, int_bits - (m % int_bits) - 1)
        comp_hash_codes[:, int(m/int_bits)] = np.bitwise_or(comp_hash_codes[:, int(m/int_bits)], tmp)
    return comp_hash_codes