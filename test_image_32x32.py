import argparse
import os
import time

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
from tqdm import tqdm

from model import Generator

parser = argparse.ArgumentParser(description='Test Multiple Images in a Folder')
parser.add_argument('--upscale_factor', default=2, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='GPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
# parser.add_argument('--input_folder', default=r"dataset\flamingo\train\reshaped_32x32",type=str, required=True, help='folder containing low resolution images')
# parser.add_argument('--output_folder', default=r"32x32_64x64",type=str, required=True, help='folder to save super resolution images')
parser.add_argument('--model_name', default='netG_epoch_4_100.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
INPUT_FOLDER  = r"dataset\flamingo\train\reshaped_32x32"
OUTPUT_FOLDER = r"32x32_64x64"
weight_path = r"pt_files\ori_64_2x_sr\netG_epoch_2_100.pth"

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load model
model = Generator(UPSCALE_FACTOR).eval()
if TEST_MODE:
    model.cuda()
    model.load_state_dict(torch.load(weight_path))
else:
    model.load_state_dict(torch.load(weight_path))

# Process all images in the input folder
image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

for image_name in tqdm(image_files, desc="Processing images"):
    image_path = os.path.join(INPUT_FOLDER, image_name)
    output_path = os.path.join(OUTPUT_FOLDER, image_name)

    image = Image.open(image_path).convert('RGB')  # Ensure image is RGB
    image = Variable(ToTensor()(image)).unsqueeze(0)
    if TEST_MODE:
        image = image.cuda()

    start = time.time()
    with torch.no_grad():
        out = model(image)
    elapsed = time.time() - start

    out_img = ToPILImage()(out[0].data.cpu())
    out_img.save(output_path)

    print(f'Processed {image_name} in {elapsed:.4f}s')
