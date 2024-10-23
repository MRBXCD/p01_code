import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class DATA(Dataset):
  def __init__(self, path):
    self.data = np.load(path)['arr_0']
    self.len = self.data.shape[0]

  def __getitem__(self, index):
    image = self.data[index]
    image = norm_image(image)
    return torch.from_numpy(image).float()

  def __len__(self):
    return self.len

class DATA_ALL(Dataset):
  def __init__(self, path):
    # self.px = np.load(path)['px']
    # self.py = np.load(path)['py']
    self.d3data = np.load(path)['d3data']
    self.idt = np.load(path)['idt']
    self.idb = np.load(path)['idb']
    self.len = self.d3data.shape[0]

  def __getitem__(self, index):
    # px = self.px[index]
    # py = self.py[index]
    image = self.d3data[index]
    image = norm_image(image)
    idt = self.idt[index]
    idb = self.idb[index]
    return  torch.from_numpy(image).float() ,torch.from_numpy(idt).float(), torch.from_numpy(idb).float()

  def __len__(self):
    return self.len



def norm_image(image):
        min_v = image.min()
        max_v = image.max()
        return ((image - min_v) / (max_v - min_v))
