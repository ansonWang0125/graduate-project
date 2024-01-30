import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
import math
import numpy as np
from tqdm import tqdm
from UDet_4layer import UDet

    
class SliceSegmentationDataset(Dataset):
    def __init__(self, ct_slices, contextSlices_count):
        super(SliceSegmentationDataset, self).__init__()       
        self.ct_slices = ct_slices
        self.contextSlices_count = contextSlices_count

    def __len__(self):
        return 1

    def __getitem__(self, ndx): #for validation
        return self.getitem_fullSlice(ndx)

    def getitem_fullSlice(self, ndx):
        ct_t = torch.zeros((self.contextSlices_count * 2 + 1, 512, 512))  #預設是上下兩張

        start_ndx = ndx - self.contextSlices_count
        end_ndx = ndx + self.contextSlices_count + 1
        for i, context_ndx in enumerate(range(start_ndx, end_ndx)):
            context_ndx = max(context_ndx, 0) #避免邊界，遇到邊界會重複
            context_ndx = min(context_ndx, self.ct_slices.shape[0] - 1)
            ct_t[i] = torch.from_numpy(self.ct_slices[context_ndx].astype(np.float32))

        ct_t.clamp_(-1000, 1000)

        return ct_t

class SegmentationDataset(Dataset):
    def __init__(self, ct_scan, contextSlices_count):
        super(SegmentationDataset, self).__init__()       
        self.ct_scan = ct_scan
        self.contextSlices_count = contextSlices_count

    def __len__(self):
        return len(self.ct_scan)

    def __getitem__(self, ndx): #for validation
        return self.getitem_fullSlice(ndx)

    def getitem_fullSlice(self, ndx):
        ct_t = torch.zeros((self.contextSlices_count * 2 + 1, 512, 512))  #預設是上下兩張

        start_ndx = ndx - self.contextSlices_count
        end_ndx = ndx + self.contextSlices_count + 1
        for i, context_ndx in enumerate(range(start_ndx, end_ndx)):
            context_ndx = max(context_ndx, 0) #避免邊界，遇到邊界會重複
            context_ndx = min(context_ndx, self.ct_scan.shape[0] - 1)
            ct_t[i] = torch.from_numpy(self.ct_scan[context_ndx].astype(np.float32))

        ct_t.clamp_(-1000, 1000)

        return ct_t
    
class UDetWrapper(nn.Module):
    def __init__(self, **kwargs): #kwarg is a dictionary containing all keyword arguments passed to the constructor
        super().__init__()

        # we will do batchnormalization first 
        self.input_batchnorm = nn.BatchNorm2d(kwargs['in_channels'])  #in kwarg, we have in_channels params to give the input channel
        self.udet = UDet(**kwargs)
        self.final = nn.Sigmoid() #use sigmoid to limit the output to 0,1

        self._init_weights()

    def _init_weights(self):
        init_set = {
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
            nn.Linear,
        }
        for m in self.modules():
            if type(m) in init_set:
                nn.init.kaiming_normal_(
                    m.weight.data, mode='fan_out', nonlinearity='relu', a=0
                )
                if m.bias is not None:
                    fan_in, fan_out = \
                        nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

        # nn.init.constant_(self.unet.last.bias, -4)
        # nn.init.constant_(self.unet.last.bias, 4)


    def forward(self, input_batch):
        bn_output = self.input_batchnorm(input_batch)
        un_output = self.udet(bn_output)
        fn_output = self.final(un_output)
        return fn_output
    
    
def nodule_segment(ct_scan):
    segmentation_model = UDetWrapper(
            in_channels=7,
            n_classes=1,
            depth=4,  #how deep the U go
            wf=6,   #2^4 filter
            padding=True, #padding so that we get the output size as input size
            batch_norm=True,
            up_mode='upconv', #use  nn.ConvTranspose2d
        )
    # model_state
    # torch.load("F:\\udet\\models\\udet\\seg_2023-10-19_08.28.18_final-cls.best.state")["model_state"]
    # segmentation_model.load_state_dict(torch.load("F:\\udet\\models\\udet\\u_net_depth2_200epcoch_f1score0.2.state")["model_state"])
    segmentation_model.load_state_dict(torch.load("C://LUNA//udet//models//udet//best-model//augseg_new2.state")["model_state"])
    device = torch.device("cuda")
    segmentation_model.to(device)
    segmentation_model.eval()
    test_ds = SegmentationDataset(ct_scan, contextSlices_count=3)

    batch_size = 2

    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=16,
        pin_memory=True,
    )
    slicesize = len(ct_scan)
    with torch.no_grad():
        pred_mat = []
        for ct_t in tqdm(test_dl, desc="Predict"):
            test_pred = segmentation_model(ct_t.to(device))
            pred_mat.append(test_pred)
        pred_np = np.empty((slicesize, 512, 512))
        for index, tensor in enumerate(pred_mat):
            tensor_len = len(tensor)
            pred_np[2*index:2*index + tensor_len] = tensor.cpu().numpy().reshape(tensor_len, 512, 512)
            
    return pred_np

def slice_segment(ct_slices):
    segmentation_model = UDetWrapper(
            in_channels=7,
            n_classes=1,
            depth=4,  #how deep the U go
            wf=6,   #2^4 filter
            padding=True, #padding so that we get the output size as input size
            batch_norm=True,
            up_mode='upconv', #use  nn.ConvTranspose2d
        )
    segmentation_model.load_state_dict(torch.load("C://LUNA//udet//models//udet//best-model//augseg_new2.state")["model_state"])
    device = torch.device("cuda")
    segmentation_model.to(device)
    segmentation_model.eval()
    test_ds = SliceSegmentationDataset(ct_slices, contextSlices_count=3)

    batch_size = 1

    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=16,
        pin_memory=True,
    )
    slicesize = 1
    with torch.no_grad():
        pred_mat = []
        for ct_t in test_dl:
            test_pred = segmentation_model(ct_t.to(device))
            pred_mat.append(test_pred)
        pred_np = np.empty((slicesize, 512, 512))
        for index, tensor in enumerate(pred_mat):
            pred_np[index] = tensor.cpu().numpy().reshape(512, 512)
            
    return pred_np
            
            