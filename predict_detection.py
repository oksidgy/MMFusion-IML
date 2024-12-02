import argparse
import logging
import time
from os import listdir
from os.path import isfile, join

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from albumentations.pytorch import ToTensorV2
import albumentations as A
from matplotlib import pyplot as plt
from torch.onnx.symbolic_opset9 import unsqueeze
from tqdm import tqdm

from configs.cmnext_init_cfg import _C as config, update_config
from model.cmnext_conf import CMNeXtWithConf
from model.modal_extract import ModalitiesExtractor
from model.ws_cmnext_conf import WSCMNeXtWithConf

parser = argparse.ArgumentParser(description='Test Detection')
parser.add_argument('-gpu', '--gpu', type=int, default=-1, help='device, use -1 for cpu')
parser.add_argument('-log', '--log', type=str, default='INFO', help='logging level')
parser.add_argument('-exp', '--exp', type=str, default="./experiments/ec_example_phase_earlyfusion.yaml", help='Yaml experiment file')
parser.add_argument('-ckpt', '--ckpt', type=str, default="./ckpt/early_fusion_detection.pth", help='Checkpoint')
parser.add_argument('-input', '--input', type=str, default="./data/test", help='Directory of input files')

#  parser.add_argument('opts', help="other options", default=None, nargs=argparse.REMAINDER)

args = parser.parse_args()

config = update_config(config, args.exp)

gpu = args.gpu
loglvl = getattr(logging, args.log.upper())
logging.basicConfig(level=loglvl)

device = 'cuda:%d' % gpu if gpu >= 0 else 'cpu'
np.set_printoptions(formatter={'float': '{: 7.3f}'.format})
target = "./results/maps"

if device != 'cpu':
    # cudnn setting
    import torch.backends.cudnn as cudnn

    cudnn.benchmark = False
    cudnn.deterministic = True
    cudnn.enabled = config.CUDNN.ENABLED


modal_extractor = ModalitiesExtractor(config.MODEL.MODALS[1:], config.MODEL.NP_WEIGHTS)

def predict():
    if args.ckpt.endswith("early_fusion_detection.pth"):
        model = CMNeXtWithConf(config.MODEL)
    else:
        model = WSCMNeXtWithConf(config.MODEL)

    ckpt = torch.load(args.ckpt, map_location=device)

    model.load_state_dict(ckpt['state_dict'])
    modal_extractor.load_state_dict(ckpt['extractor_state_dict'])

    modal_extractor.to(device)
    model = model.to(device)
    modal_extractor.eval()
    model.eval()

    times = []
    image_paths = [join(args.input, f) for f in listdir(args.input) if isfile(join(args.input, f))]

    transforms = []

    # transforms.append(A.LongestMaxSize(max_size=2048))
    transforms.append(ToTensorV2())
    image_transforms_final = A.Compose(transforms)

    for image_path in tqdm(image_paths):
        with torch.no_grad():
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            h, w, c = image.shape
            if h > 2048 or w > 2048:
                res = A.LongestMaxSize(max_size=2048)(image=image)
                image = res['image']
            image = image_transforms_final(image=image)['image']

            image = image.to(device, non_blocking=True)
            image = image / 256.0
            batch = torch.unsqueeze(image, 0)

            modals = modal_extractor(batch)

            time_b = time.time()

            images_norm = TF.normalize(batch, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            inp = [images_norm] + modals

            anomaly, confidence, detection = model(inp)
            detection = torch.sigmoid(detection)
            det = detection.squeeze().cpu().item()
            map = torch.nn.functional.softmax(anomaly, dim=1)[:, 1, :, :].squeeze().cpu().numpy()
            plt.imsave(join(target, image_path.split('/')[-1]), map, cmap='RdBu_r', vmin=0, vmax=1)

            print(f"Image {image_path.split('/')[-1]} is detection={det}, is_manip={det > 0.5}")

            times.append(time.time() - time_b)

    print(f"\nTime predict (avg): {np.array(times).mean()}")


if __name__ == "__main__":
    predict()