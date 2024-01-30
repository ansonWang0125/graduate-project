import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# import models as M
import numpy as np
import scipy
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from scipy.ndimage.morphology import binary_erosion, binary_fill_holes
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from IPython.display import HTML
from base64 import b64encode
import SimpleITK as sitk
import skimage, skimage.morphology, skimage.data
import copy
import imageio.v3 as iio
import random
random.seed(42)

class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, activation, frame_size):
        super(ConvLSTMCell, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')

        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu

        self.conv = nn.Conv2d(
            in_channels=in_channels + out_channels,
            out_channels=4 * out_channels,
            kernel_size=kernel_size,
            padding=padding)

        self.W_ci = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_co = nn.Parameter(torch.Tensor(out_channels, *frame_size))
        self.W_cf = nn.Parameter(torch.Tensor(out_channels, *frame_size))

        # Initialize weights using Xavier initialization
        nn.init.xavier_uniform_(self.W_ci)
        nn.init.xavier_uniform_(self.W_co)
        nn.init.xavier_uniform_(self.W_cf)

    def forward(self, X, H_prev, C_prev):
        # print(X.shape, H_prev.shape)
        conv_output = self.conv(torch.cat([X, H_prev], dim=1))
        i_conv, f_conv, C_conv, o_conv = torch.chunk(conv_output, chunks=4, dim=1)

        input_gate = torch.sigmoid(i_conv + self.W_ci * C_prev)
        forget_gate = torch.sigmoid(f_conv + self.W_cf * C_prev)

        # Current Cell output
        C = forget_gate * C_prev + input_gate * self.activation(C_conv)

        output_gate = torch.sigmoid(o_conv + self.W_co * C)

        # Current Hidden State
        H = output_gate * self.activation(C)

        return H, C


class ConvLSTM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, activation, frame_size, return_sequence=False):
        super(ConvLSTM, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self.out_channels = out_channels
        self.return_sequence = return_sequence

        # We will unroll this over time steps
        self.convLSTMcell = ConvLSTMCell(in_channels, out_channels, kernel_size, padding, activation, frame_size)

    def forward(self, X):
        # X is a frame sequence (batch_size, seq_len, num_channels, height, width)

        # Get the dimensions
        batch_size, seq_len, channels, height, width = X.size()

        # Initialize output
        output = torch.zeros(batch_size, seq_len, self.out_channels, height, width, device=self.device)

        # Initialize Hidden State
        H = torch.zeros(batch_size, self.out_channels, height, width, device=self.device)

        # Initialize Cell Input
        C = torch.zeros(batch_size, self.out_channels, height, width, device=self.device)

        # Unroll over time steps
        for time_step in range(seq_len):
            H, C = self.convLSTMcell(X[:, time_step, ...], H, C)
            # H, C = self.convLSTMcell(X, H, C)
            output[:, time_step, ...] = H

        if not self.return_sequence:
            output = torch.squeeze(output[:, -1, ...], dim=1)

        return output

class ConvBLSTM(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, padding, activation, frame_size, return_sequence=False):
        super(ConvBLSTM, self).__init__()
        self.return_sequence = return_sequence
        self.forward_cell = ConvLSTM(in_channels, out_channels//2, 
                                     kernel_size, padding, activation, frame_size, return_sequence=True)
        self.backward_cell = ConvLSTM(in_channels, out_channels//2, 
                                     kernel_size, padding, activation, frame_size, return_sequence=True)

    def forward(self, x):
        y_out_forward = self.forward_cell(x)
        reversed_idx = list(reversed(range(x.shape[1])))
        y_out_reverse = self.backward_cell(x[:, reversed_idx, ...])[:, reversed_idx, ...]
        output = torch.cat((y_out_forward, y_out_reverse), dim=2)
        if not self.return_sequence:
            output = torch.squeeze(output[:, -1, ...], dim=1)
        return output

class BCDUNet(nn.Module):
    def __init__(self, input_dim=3, output_dim=3, num_filter=64, frame_size=(256, 256), bidirectional=False, norm='instance'):
        super(BCDUNet, self).__init__()
        self.num_filter = num_filter
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)
        self.frame_size = np.array(frame_size)

        if norm == 'instance':
            norm_layer = nn.InstanceNorm2d
        else:
            norm_layer = nn.BatchNorm2d

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                norm_layer(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                norm_layer(out_channels),
                nn.ReLU(inplace=True)
            )
        
        # Encoder
        resnet34 = models.resnet34(pretrained=True)
        filters = [64, 128, 256, 512]
        
        self.res_input = resnet34.conv1
        self.res_bn1 = nn.BatchNorm2d(64)
        self.res_bn2 = nn.BatchNorm2d(128)
        self.res_bn3 = nn.BatchNorm2d(256)
        self.res_bn4 = nn.BatchNorm2d(512)
        self.res_relu = nn.ReLU(inplace=False)
        self.res_maxpool = resnet34.maxpool
        self.encoder1 = resnet34.layer2
        self.encoder2 = resnet34.layer3
        self.encoder3 = resnet34.layer4

        self.bridge = nn.Sequential(
            nn.Conv2d(filters[3], filters[3]*2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters[3]*2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            
        )

        self.upconv3 = nn.ConvTranspose2d(num_filter * 8, num_filter * 4, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(num_filter * 4, num_filter * 2, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(num_filter * 2, num_filter, kernel_size=2, stride=2)
        self.upconv0 = nn.ConvTranspose2d(num_filter, output_dim, kernel_size=2, stride=2)

        self.conv3m = conv_block(num_filter * 8, num_filter * 4)
        self.conv2m = conv_block(num_filter * 4, num_filter * 2)
        self.conv1m = conv_block(num_filter * 2, num_filter)

        self.conv0 = nn.Conv2d(output_dim, output_dim, kernel_size=1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        if bidirectional:
            self.clstm1 = ConvBLSTM(num_filter*4, num_filter*2, (3, 3), (1,1), 'tanh', list(self.frame_size//4))
            self.clstm2 = ConvBLSTM(num_filter*2, num_filter, (3, 3), (1,1), 'tanh', list(self.frame_size//2))
            self.clstm3 = ConvBLSTM(num_filter, num_filter//2, (3, 3), (1,1), 'tanh', list(self.frame_size))
        else:
            self.clstm1 = ConvLSTM(num_filter*4, num_filter*2, (3, 3), (1,1), 'tanh', list(self.frame_size//4))
            self.clstm2 = ConvLSTM(num_filter*2, num_filter, (3, 3), (1,1), 'tanh', list(self.frame_size//2))
            self.clstm3 = ConvLSTM(num_filter, num_filter//2, (3, 3), (1,1), 'tanh', list(self.frame_size))

    def forward(self, x):
        N = self.frame_size

        ## Encoder 
        conv1 = self.res_input(x)
        conv1 = self.res_relu(conv1)
        conv1 = self.res_bn1(conv1)
        conv2 = self.encoder1(conv1)
        conv2 = self.res_relu(conv2)
        conv2 = self.res_bn2(conv2)
        conv3 = self.encoder2(conv2)
        conv3 = self.res_bn3(conv3)
        conv4 = self.encoder3(conv3)
        conv4 = self.res_bn4(conv4)

        ## Decoder
        upconv3 = self.upconv3(conv4)
        upconv32 = upconv3.unsqueeze(0).transpose(0, 1)
        upconv32 = torch.cat([upconv32] * 2, dim=1)
        concat3 = self.clstm1(upconv32)
        concat3 = torch.cat((concat3, concat3), 1)
        concat3 = torch.cat((conv3, concat3), 1)
        conv3m = self.conv3m(concat3)
        conv3m = self.relu(conv3m)

        upconv2 = self.upconv2(conv3m)
        upconv22 = upconv2.unsqueeze(0).transpose(0, 1)
        upconv22 = torch.cat([upconv22] * 2, dim=1)
        concat2 = self.clstm2(upconv22)
        concat2 = torch.cat((concat2, concat2), 1)
        concat2 = torch.cat((conv2, concat2), 1)
        conv2m = self.conv2m(concat2)
        conv2m = self.relu(conv2m)

        upconv1 = self.upconv1(conv2m)
        upconv12 = upconv1.unsqueeze(0).transpose(0, 1)
        upconv22 = torch.cat([upconv22] * 2, dim=1)
        concat1 = self.clstm3(upconv12)
        concat1 = torch.cat((concat1, concat1), 1)
        concat1 = torch.cat((conv1, concat1), 1)
        conv1m = self.conv1m(concat1)
        conv1m = self.relu(conv1m)

        upconv0 = self.upconv0(conv1m)
        conv0 = self.conv0(upconv0)

        return conv0

    
from torch.utils.data import Dataset
import numpy as np
np.random.seed(42)

def train_valid_split(data_set, valid_ratio, seed=42):
    '''Split provided training data into training set and validation set'''
    valid_set_size = int(valid_ratio * len(data_set)) 
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)

class Lung_Dataset(Dataset):
    '''
    x: Features.
    y: Targets, if none, do prediction.
    '''
    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)
    
    from collections import deque
from PIL import Image
from tqdm import tqdm

def predict(test_loader, model, device):
        model.eval() # Set your model to evaluation mode.
        preds = []
        pbar = tqdm(range(len(test_loader)))
        pbar.set_description("Predicting")
        for x in test_loader:
            x = x.to(device)                        
            with torch.no_grad():                   
                pred = model(x)                     
                preds.append(pred.detach().cpu())  
                pbar.update(1) 
        preds = torch.cat(preds, dim=0).numpy()  
        return preds
    
def segmentation(te_data):
    amount = len(te_data)
    result = []
    te_data  = np.expand_dims(te_data, axis=3)
    te_data2 = te_data / 255
    te_data2 = torch.tensor(te_data2).transpose(1, 3)
    te_data2 = torch.cat([te_data2] * 3, dim=1).numpy()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Hyperparameters
    input_dim = 3
    output_dim = 3
    num_filter = 64
    frame_size = (256, 256)
    bidirectional = True
    norm = 'instance'
    batch_size = 2
    test_dataset = Lung_Dataset(te_data2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)\

    # Evaluation loop
    model = BCDUNet(input_dim, output_dim, num_filter, frame_size, bidirectional, norm).to(device)
    model.load_state_dict(torch.load('model.pth'))
    # model.load_state_dict(torch.load('model.pth'))
    predictions = predict(test_loader, model, device) 

    # Post-processing
    predictions = np.squeeze(predictions)
    predictions = torch.tensor(predictions).transpose(1, 3)
    predictions = np.where(predictions<0.5, 1, 0)
    Estimated_lung = predictions[:,:,:,0]
    Estimated_lung2 = copy.deepcopy(Estimated_lung)

    Estimated_lung, Estimated_lung2, Lung_mask = hole_filler(Estimated_lung, Estimated_lung2)

    Filled_Lung = copy.deepcopy(Lung_mask)
    for k in tqdm(range(Filled_Lung.shape[0]), desc="Second phase filling"):
        Filled_Lung[k] = scipy.ndimage.binary_dilation(Filled_Lung[k], iterations=5)
        Filled_Lung[k] = scipy.ndimage.binary_erosion(Filled_Lung[k], iterations=5)
        noFill = np.zeros((512, 512))
        visited = np.zeros((512, 512))
        queue = deque([(0, 0)])
        while queue:
            node = queue.popleft()
            if visited[node[0]][node[1]] == 0:
                visited[node[0]][node[1]] = 1
                noFill[node[0]][node[1]] = 1
                for d in [(-1, 0), (0, 1), (0, -1), (1, 0)]:
                    if node[0]+d[0] >= 0 and node[0]+d[0] < 512 and node[1]+d[1] >= 0 and node[1]+d[1] < 512:
                        if Filled_Lung[k][node[0]+d[0]][node[1]+d[1]] == 0:
                            queue.append((node[0]+d[0], node[1]+d[1]))
        for i in range(512):
            for j in range(512):
                if noFill[i][j] != 1:
                    Filled_Lung[k][i][j] = 1

    amount = len(te_data)
    for idx in tqdm(range(amount), desc="Computing segmentation result"):
        seg_result =  np.squeeze(te_data[idx])*Filled_Lung[idx]
        seg_result = seg_result.astype(int)
        seg_result[seg_result == 0] = 1000
        result.append(seg_result)
    result = np.stack(result, axis=0)
    return result


def edge_clean(matrix):
    for i in range(0, 5):
        for j in range(0, 512):
            matrix[i][j] = 0
            matrix[j][i] = 0
    for i in range(507, 512):
        for j in range(0, 512):
            matrix[i][j] = 0
            matrix[j][i] = 0

def hole_filler(Estimated_lung, Estimated_lung2):
    for k in tqdm(range(Estimated_lung.shape[0]), desc="First phase filling"):
        edge_clean(Estimated_lung[k])
        edge_clean(Estimated_lung2[k])
        Estimated_lung[k] = scipy.ndimage.binary_erosion(Estimated_lung[k], iterations=5)
        Estimated_lung2[k] = scipy.ndimage.binary_erosion(Estimated_lung2[k], iterations=5)
        noFill = np.zeros((512, 512))
        visited = np.zeros((512, 512))
        queue = deque([(0, 0)])
        while queue:
            node = queue.popleft()
            if visited[node[0]][node[1]] == 0:
                visited[node[0]][node[1]] = 1
                noFill[node[0]][node[1]] = 1
                for d in [(-1, 0), (0, 1), (0, -1), (1, 0)]:
                    if node[0]+d[0] >= 0 and node[0]+d[0] < 512 and node[1]+d[1] >= 0 and node[1]+d[1] < 512:
                        if Estimated_lung[k][node[0]+d[0]][node[1]+d[1]] == 0:
                            queue.append((node[0]+d[0], node[1]+d[1]))
        for i in range(512):
            for j in range(512):
                if noFill[i][j] != 1:
                    Estimated_lung[k][i][j] = 1
    Lung_mask = np.subtract(Estimated_lung, Estimated_lung2)
    return Estimated_lung, Estimated_lung2, Lung_mask