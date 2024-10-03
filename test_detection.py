import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
import logging
import torch
import torchvision.transforms.functional as TF
import time

from data.datasets import ManipulationDataset, PreprocessingType
from common.metrics import computeDetectionMetrics

from model.cmnext_conf import CMNeXtWithConf
from model.modal_extract import ModalitiesExtractor
from configs.cmnext_init_cfg import _C as config, update_config
from model.ws_cmnext_conf import WSCMNeXtWithConf

parser = argparse.ArgumentParser(description='Test Detection')
parser.add_argument('-gpu', '--gpu', type=int, default=-1, help='device, use -1 for cpu')
parser.add_argument('-log', '--log', type=str, default='INFO', help='logging level')
parser.add_argument('-exp', '--exp', type=str, default="./experiments/ec_example_phase_earlyfusion.yaml", help='Yaml experiment file')
parser.add_argument('-ckpt', '--ckpt', type=str, default="./ckpt/early_fusion_detection.pth", help='Checkpoint')
parser.add_argument('-manip', '--manip', type=str, default="./data/IDT-DSO-1-manip.txt", help='Manip data file')
parser.add_argument('-auth', '--auth', type=str, default="./data/IDT-DSO-1-auth.txt", help='Auth data file')
parser.add_argument('opts', help="other options", default=None, nargs=argparse.REMAINDER)

args = parser.parse_args()

config = update_config(config, args.exp)

gpu = args.gpu
loglvl = getattr(logging, args.log.upper())
logging.basicConfig(level=loglvl)

device = 'cuda:%d' % gpu if gpu >= 0 else 'cpu'
np.set_printoptions(formatter={'float': '{: 7.3f}'.format})

if device != 'cpu':
    # cudnn setting
    import torch.backends.cudnn as cudnn

    cudnn.benchmark = False
    cudnn.deterministic = True
    cudnn.enabled = config.CUDNN.ENABLED


modal_extractor = ModalitiesExtractor(config.MODEL.MODALS[1:], config.MODEL.NP_WEIGHTS)

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

preprocessing = PreprocessingType.compress

manip = ManipulationDataset(args.manip,
                            config.DATASET.IMG_SIZE,
                            train=False,
                            preprocessing=None)
auth = ManipulationDataset(args.auth,
                           config.DATASET.IMG_SIZE,
                           train=False,
                           preprocessing=None)
val = ConcatDataset([manip, auth])
val_loader = DataLoader(val,
                        batch_size=1,
                        shuffle=True,
                        num_workers=config.WORKERS,
                        pin_memory=True)

scores = []
labels = []
times = []
pbar = tqdm(val_loader)
for step, (images, _, masks, lab) in enumerate(pbar):
    with torch.no_grad():
        images = images.to(device, non_blocking=True)
        masks = masks.squeeze(1).to(device, non_blocking=True)
        lab = lab.to(device, non_blocking=True)
        modals = modal_extractor(images)

        time_b = time.time()

        images_norm = TF.normalize(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        inp = [images_norm] + modals

        anomaly, confidence, detection = model(inp)

        detection = torch.sigmoid(detection)

        time_batch = (time.time() - time_b) / len(images)
        scores.append(detection.squeeze().cpu().item())
        labels.append(lab.squeeze().cpu().item())
        times.append(time_batch)

print(f"\nTime predict (avg): {np.array(times).mean()}")
auc, baCC, tn, fp, fn, tp, pr, rec, f1 = computeDetectionMetrics(scores, labels)
print("\nAUC: {}\nbACC: {}".format(auc, baCC))
print(f"\nTP={tp}, TN={tn}, FP={fp}, FN={fn}")
print(f"\nManip: pre={pr[1]}, recall={rec[1]}, F1={f1[1]}")
print(f"\nNot Manip: pre={pr[0]}, recall={rec[0]}, F1={f1[0]}")
