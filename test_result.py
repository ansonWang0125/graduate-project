from util import logging, enumerateWithEstimate
import torch
import numpy as np
import matplotlib.pyplot as plt
import copy
import random
import torch.nn.functional as F
import time
import argparse
import datetime
import sys
from torch.utils.data import DataLoader
import math
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import shutil
import hashlib
from torch import nn as nn
from torch.utils.data import Dataset
import os
sys.path.append('./')
from dsetsFullCT import TrainingLuna2dSegmentationDataset, Luna2dSegmentationDataset, PrepcacheLunaDataset, getCt
from util import logging, enumerateWithEstimate
from UDet_4layer import UDet
from metrics import dc, jc, sensitivity, specificity, precision, recall
import csv
import SimpleITK as sitk
from skimage import measure, filters
import scipy.ndimage.morphology
from os.path import join
import matplotlib.patches as mpatches

def threshold_mask(raw_output, threshold):
    if threshold == 0:
        try:
            threshold = filters.threshold_otsu(raw_output)
        except:
            threshold = 0.5

    # print('\tThreshold: {}'.format(threshold))

    raw_output[raw_output > threshold] = 1
    raw_output[raw_output < 1] = 0

    all_labels = measure.label(raw_output)
    props = measure.regionprops(all_labels)
    props.sort(key=lambda x: x.area, reverse=True)
    thresholded_mask = np.zeros(raw_output.shape)

    if len(props) >= 2:
        if props[0].area / props[1].area > 5:  # if the largest is way larger than the second largest
            thresholded_mask[all_labels == props[0].label] = 1  # only turn on the largest component
        else:
            thresholded_mask[all_labels == props[0].label] = 1  # turn on two largest components
            thresholded_mask[all_labels == props[1].label] = 1
    elif len(props):
        thresholded_mask[all_labels == props[0].label] = 1

    thresholded_mask = scipy.ndimage.morphology.binary_fill_holes(thresholded_mask).astype(np.uint8)

    return thresholded_mask

class WeightedDiceLoss(nn.Module):
    def __init__(self, weight=None):
        super(WeightedDiceLoss, self).__init__()
        self.weight = weight

    def forward(self, input, target):
        smooth = 1.  # Smoothing factor to prevent division by zero

        iflat = input.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()

        if self.weight is not None:
            # Apply class-wise weights
            dice = 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))
            weighted_dice = dice * self.weight
            return weighted_dice.mean()
        else:
            return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))
        
def plot_image(data, batch_ndx, output, outdir, index):
    # print(data.size())
    output_cpu = output.cpu().squeeze(1)
    img_name = "{}_{}".format(data[2][batch_ndx], index)
    img_data = data[0][batch_ndx].squeeze().numpy()
    gt_data = data[1][batch_ndx].squeeze(1).numpy()
    output_bin = threshold_mask(output_cpu, 0.5)
    fig_out_dir = "C://LUNA//udet//result//{}//qual_figs".format(outdir)
    f, ax = plt.subplots(figsize=(10, 5))

    ax.imshow(img_data[img_data.shape[0] // 2 , :, :], cmap='bone')
    ax.imshow(output_bin[0, :, :], cmap='Blues', alpha=0.5, vmin=0, vmax=1)
    ax.imshow(gt_data[0, :, :], cmap='Reds', alpha=0.5, vmin=0, vmax=1)
    contour_pred = ax.contour(output_bin[0, :, :], levels=[0.5], colors='red', linewidths=1)
    contour_truth = ax.contour(gt_data[0, :, :], levels=[0.5], colors='blue', linewidths=1)
    # Create legend
    legend_pred = mpatches.Patch(color='red', label='Predicted Mask')
    legend_truth = mpatches.Patch(color='blue', label='Ground Truth Mask')
    ax.legend(handles=[legend_pred, legend_truth])
    ax.axis('off')
    fig = plt.gcf()
    fig.suptitle(img_name)

    # print("save image at: ", join(fig_out_dir, img_name + '_qual_fig' + '.png'))
    plt.savefig(join(fig_out_dir, img_name + '_qual_fig' + '.png'),
                format='png', bbox_inches='tight')
    plt.close('all')

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

METRICS_LOSS_NDX = 1
METRICS_FNLOSS_NDX = 2
METRICS_FPLOSS_NDX = 3
METRICS_TP_NDX = 7
METRICS_FN_NDX = 8
METRICS_FP_NDX = 9

METRICS_SIZE = 10  

def write_csv_header(csv_writer):
    header = ["UID", "NDX", "Dice Loss", "Dice", "Jaccard", "Precision", "Recall", "Sensitivity", "Specificity"]
    csv_writer.writerow(header)

def write_metrics_to_csv(writer, uid, slice_ndx, dloss, dice, jaccard, precision, recall, sensitivity, specificity):
    row = [uid, slice_ndx, dloss, dice, jaccard, precision, recall, sensitivity, specificity]
    writer.writerow(row)

def doTesting(epoch_ndx, test_dl, segmentation_model, device, csv_filename, outdir, allow_plot, isAug):
    with torch.no_grad():
        testMetrics_g = torch.zeros(METRICS_SIZE, len(test_dl.dataset), device=device)
        batch_iter = enumerateWithEstimate(
            test_dl,
            "E{} Testing ".format(epoch_ndx),
            start_ndx=test_dl.num_workers,
        )

        with open(csv_filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            write_csv_header(csv_writer)

            for batch_ndx, batch_tup in batch_iter:
                computeBatchLoss(batch_ndx, batch_tup, test_dl.batch_size, testMetrics_g, segmentation_model, device, csv_writer, outdir, 0.5, allow_plot, isAug)
                # break

    return testMetrics_g.to('cpu')

def computeBatchLoss(batch_ndx, batch_tup, batch_size, metrics_g, segmentation_model, device, csv_writer, outdir,
                     classificationThreshold=0.5, allow_plot=False, isAug = False):
    input_t, label_t, series_list, _slice_ndx_list = batch_tup

    input_g = input_t.to(device, non_blocking=True)
    label_g = label_t.to(device, non_blocking=True)
    prediction_g = segmentation_model(input_g)
    # print(label_g.size())
    # print(prediction_g.size())
    batch_n = len(series_list)
    if allow_plot:
        for i in range(batch_n):
            # print(_slice_ndx_list[i].item())
            # print(_slice_ndx_list[i].item() % 2 == 0)
            if _slice_ndx_list[i].item() % 2 == 0 or not isAug:
                index = int(_slice_ndx_list[i].item() if not isAug else (_slice_ndx_list[i].item()) / 2)
                # print(index)
                # print((_slice_ndx_list[i].item() + 1) / 2)
                torch.save({'label_g': label_g[i], 'prediction_g': prediction_g[i]}, 'C://LUNA//udet//result//{}//output//{}_{}.pth'.format(outdir, series_list[i], index))
                plot_image(batch_tup, i, prediction_g[i], outdir, index)
    # pos_weight = torch.tensor([100]).to(device, non_blocking=True)
    # criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion = WeightedDiceLoss()
    # BCELoss = criterion(prediction_g, label_g.to(torch.float))
    fpLoss_g = criterion(prediction_g * ~label_g, label_g.to(torch.float))
    fnLoss_g = criterion(prediction_g * label_g, label_g.to(torch.float))
    diceloss = criterion(prediction_g, label_g.to(torch.float))

    # Additional metric calculations
    if allow_plot:
        for i in range(batch_n):
            if _slice_ndx_list[i].item() % 2 == 0 or not isAug:
                index = int(_slice_ndx_list[i].item() if not isAug else (_slice_ndx_list[i].item()) / 2)
                output_bin = (prediction_g[i] > classificationThreshold).to(torch.float32).cpu().numpy()
                gt_data = label_g[i].cpu().numpy()

                dice = dc(output_bin, gt_data)
                jaccard = jc(output_bin, gt_data)
                precision_val = precision(output_bin, gt_data)
                recall_val = recall(output_bin, gt_data)
                sensitivity_val = sensitivity(output_bin, gt_data)
                specificity_val = specificity(output_bin, gt_data)
                # assd_val = assd(output_bin, gt_data, voxelspacing=sitk_img.GetSpacing(), connectivity=1)

                # Write metrics to CSV
                write_metrics_to_csv(csv_writer, series_list[i], index, diceloss.item(), dice, jaccard, precision_val, recall_val, sensitivity_val, specificity_val)

    start_ndx = batch_ndx * batch_size
    end_ndx = start_ndx + input_t.size(0)

    # total_loss = BCELoss + fpLoss_g.mean()
    total_loss = diceloss.mean()

    with torch.no_grad():
            predictionBool_g = (prediction_g[:, 0:1]
                                > classificationThreshold).to(torch.float32)

            tp = (     predictionBool_g *  label_g).sum(dim=[1,2,3])
            fn = ((1 - predictionBool_g) *  label_g).sum(dim=[1,2,3])
            fp = (     predictionBool_g * (~label_g)).sum(dim=[1,2,3])

            # metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = BCELoss
            metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = total_loss
            # metrics_g[METRICS_FNLOSS_NDX, start_ndx:end_ndx] = fnLoss_g
            metrics_g[METRICS_FPLOSS_NDX, start_ndx:end_ndx] = fpLoss_g
            metrics_g[METRICS_TP_NDX, start_ndx:end_ndx] = tp
            metrics_g[METRICS_FN_NDX, start_ndx:end_ndx] = fn
            metrics_g[METRICS_FP_NDX, start_ndx:end_ndx] = fp

    return total_loss
    
def logMetrics(epoch_ndx, mode_str, metrics_t):
    log = logging.getLogger(__name__)
    log.info("E{}".format(
        epoch_ndx,
    ))

    metrics_a = metrics_t.detach().numpy()
    # print(metrics_a.shape)
    sum_a = metrics_a.sum(axis=1)
    # print(sum_a.shape)
    assert np.isfinite(metrics_a).all()

    allLabel_count = sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]
    # print("all label count: ", allLabel_count)

    metrics_dict = {}
    metrics_dict['loss/all'] = metrics_a[METRICS_LOSS_NDX].mean()
    # metrics_dict['fnloss/all'] = metrics_a[METRICS_FNLOSS_NDX].mean()
    metrics_dict['fploss/all'] = metrics_a[METRICS_FPLOSS_NDX].mean()   

    metrics_dict['percent_all/tp'] = \
        sum_a[METRICS_TP_NDX] / (allLabel_count or 1) * 100
    metrics_dict['percent_all/fn'] = \
        sum_a[METRICS_FN_NDX] / (allLabel_count or 1) * 100
    metrics_dict['percent_all/fp'] = \
        sum_a[METRICS_FP_NDX] / (allLabel_count or 1) * 100


    precision = metrics_dict['pr/precision'] = sum_a[METRICS_TP_NDX] \
        / ((sum_a[METRICS_TP_NDX] + sum_a[METRICS_FP_NDX]) or 1)
    recall    = metrics_dict['pr/recall']    = sum_a[METRICS_TP_NDX] \
        / ((sum_a[METRICS_TP_NDX] + sum_a[METRICS_FN_NDX]) or 1)

    metrics_dict['pr/f1_score'] = 2 * (precision * recall) \
        / ((precision + recall) or 1)


    log.info(("E{} {:8} "
            # + "CL: {}"
             + "{loss/all:.4f} loss, "
              # + "{fnloss/all:.4f} fnloss, "
              + "{fploss/all:.4f} fploss, "
             + "{pr/precision:.4f} precision, "
             + "{pr/recall:.4f} recall, "
             + "{pr/f1_score:.4f} f1 score"
              ).format(
        epoch_ndx,
        mode_str,
        **metrics_dict,
    ))
    log.info(("E{} {:8} "
              + "{loss/all:.4f} loss, "
              + "{percent_all/tp:-5.1f}% tp, {percent_all/fn:-5.1f}% fn, {percent_all/fp:-9.1f}% fp"
    ).format(
        epoch_ndx,
        mode_str + '_all',
        **metrics_dict,
    ))

    score = metrics_dict['pr/recall']

    return score
    
def test_result(allow_plot, outdir, isAug):
    segmentation_model = UDetWrapper(
                in_channels=7,
                n_classes=1,
                depth=4,  #how deep the U go
                wf=6,   #2^4 filter
                padding=True, #padding so that we get the output size as input size
                batch_norm=True,
                up_mode='upconv', #use  nn.ConvTranspose2d
            )

    # segmentation_model.load_state_dict(torch.load("C://LUNA//udet//models//udet//best-model//seg_2023-11-18_23.45.55_final-cls.best.state")["model_state"])
    segmentation_model.load_state_dict(torch.load("C://LUNA//udet//models//udet//best-model//augseg_cor_new.state")["model_state"])
    # segmentation_model.load_state_dict(torch.load("C://LUNA//udet//models//udet//best-model//augseg_new2.state")["model_state"])
    epoch_ndx = 1

    device = torch.device("cuda")
    test_ds = Luna2dSegmentationDataset(
        val_stride=10,
        set_class="Testing",
        contextSlices_count=3,
    )

    batch_size = 2
    batch_size *= torch.cuda.device_count()
    csv_filename = "C://LUNA//udet//result//{}//result.csv".format(outdir)

    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=16,
        pin_memory=True,
    )
    segmentation_model.to(device)
    testMetrics_t = doTesting(epoch_ndx, test_dl, segmentation_model, device, csv_filename, outdir, allow_plot, isAug)
    logMetrics(epoch_ndx, 'test', testMetrics_t)