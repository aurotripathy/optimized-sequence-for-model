#!/usr/bin/env python3

"""
Needs the imagnet 'val' folder with the structure retained 
and a few images for the calibration functions to work
"""
import sys
import argparse
import onnx
import torch
import torchvision
from torchvision import transforms
import tqdm
from pathlib import Path


from furiosa.optimizer import optimize_model
from furiosa.quantizer import quantize, Calibrator, CalibrationMethod

from pudb import set_trace

parser = argparse.ArgumentParser(
    prog='quantize.py',
    description='smart quantizing',)

parser.add_argument('--bs', required=True, type=int)
parser.add_argument('--model_path', required=True, type=str)
args = parser.parse_args()
print(f'Requested batch size: {args.bs}')
print(f'Requested model: {args.model_path}')


image_size = (512, 512)
BATCH_SIZE = 1
def create_quantized_dfg():

    model = onnx.load_model(args.model_path)
    preprocess = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize(image_size),
         transforms.Normalize(mean=(0.1307,), std=(0.3081,))]
    )   

    calibration_data = torchvision.datasets.ImageNet('.',
                                                     split='val',
                                                     transform=preprocess)
    calibration_dataloader = torch.utils.data.DataLoader(calibration_data,
                                          batch_size=BATCH_SIZE,
                                          shuffle=True)
    model = optimize_model(model)

    calibrator = Calibrator(model, CalibrationMethod.MIN_MAX_ASYM)

    for calibration_data, _ in tqdm.tqdm(calibration_dataloader,
                                         desc="Calibration",
                                         unit="images",
                                         mininterval=0.5):
        print(f'calibration data shape: {calibration_data.shape}')
        calibration_data = torch.permute(calibration_data, (0, 2, 3, 1))
        print(f'calibration data shape: {calibration_data.shape}')
        calibrator.collect_data([[calibration_data.numpy()]])

    ranges = calibrator.compute_range()
    print(f'ranges" {ranges}')

    model_quantized = quantize(model,
                               ranges,
                               with_quantize=False,
                               normalized_pixel_outputs=[0])
    # model_quantized = quantize(model,
    #                            ranges,)

    dfg_file_name = f"generated_{Path(args.model_path).stem}.dfg"
    with open(dfg_file_name, "wb") as f:
        f.write(bytes(model_quantized))


if __name__ == "__main__":
    sys.exit(create_quantized_dfg())    
