import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset

class DADataset(Dataset):
    def __init__(self, ct_masks, files=None):
        super(DADataset, self).__init__()
        
        self.masks = np.array(ct_masks, dtype=float)

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        slice_num = idx
        if idx > (len(self.masks) - 4):
            mask = torch.from_numpy(np.stack((self.masks[idx], self.masks[idx], self.masks[idx], self.masks[idx]), axis=0)).float()
        else: 
            mask = torch.from_numpy(np.stack((self.masks[idx], self.masks[idx + 1], self.masks[idx + 2], self.masks[idx + 3]), axis=0)).float()
        return mask, slice_num

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.down_conv1 = self.double_conv(4, 64)
        self.down_conv2 = self.double_conv(64, 128)
        self.down_conv3 = self.double_conv(128, 256)
        self.down_conv4 = self.double_conv(256, 512)
        self.up_conv1 = self.double_conv(512 + 256, 256)
        self.up_conv2 = self.double_conv(256 + 128, 128)
        self.up_conv3 = self.double_conv(128 + 64, 64)
        self.up_conv4 = nn.Conv2d(64, 1, kernel_size=1)
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Downward path
        x1 = self.down_conv1(x)
        x2 = self.maxpool(x1)
        x3 = self.down_conv2(x2)
        x4 = self.maxpool(x3)
        x5 = self.down_conv3(x4)
        x6 = self.maxpool(x5)
        x7 = self.down_conv4(x6)

        # Upward path
        x = self.upsample(x7)
        x = torch.cat([x, x5], dim=1)
        x = self.up_conv1(x)
        x = self.upsample(x)
        x = torch.cat([x, x3], dim=1)
        x = self.up_conv2(x)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up_conv3(x)
        x = self.up_conv4(x)

        return x

def aug_mask(ct_mask):
    # "cuda" only when GPUs are available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _exp_name = "R3-01_4slice_onlymask"

    # Initialize a model and put it on the specified device.
    model = UNet().to(device)

    # The number of batch size.
    batch_size = 10
    model_best = UNet().to(device)
    model_best.load_state_dict(torch.load(f"{_exp_name}_best.ckpt"))
    model_best.eval()

    batch_size = 1
    test_set =  DADataset(ct_mask)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    slicesize = len(ct_mask)
    threshold = 0.5  # You can adjust this threshold as needed
    
    with torch.no_grad():
        pred_mat = []
        orig_mat = []
        # for im, slice_num in tqdm(test_loader, desc="Prediction"):
        for im, slice_num in test_loader:
            orig_mat.append(im[:, 0, :, :])
            if slice_num[0].item() == 0:
                pred_mat.append(im[:, 0, :, :])
            if slice_num[0].item() <= float(slicesize - 4):
                test_pred = model_best(im.to(device))
                mask_testpred = (test_pred.cpu() >= threshold).to(torch.int).numpy()
                pred_mat.append(mask_testpred)
            elif slice_num[0].item() == float(slicesize) - 3:
                pass
            elif slice_num[0].item() == float(slicesize - 1):
                pass
            else:
                pred_mat.append(im[:, 0, :, :])
        aug_len = 2 * slicesize - 1
        aug_data = torch.empty((aug_len, 512 * 512), dtype=torch.int64)
        
        orig_np = np.empty((slicesize, 512, 512))
        for index, tensor in enumerate(orig_mat):
            orig_np[index] = tensor.cpu().numpy().reshape(512, 512)
            
        pred_np = np.empty((slicesize - 1, 512, 512))
        for index, npy in enumerate(pred_mat):
            pred_np[index] = npy.reshape(512, 512)
            
        final = np.empty((orig_np.shape[0] + pred_np.shape[0], 512, 512))
        final[::2] = orig_np
        final[1::2] = pred_np  
        
    return final