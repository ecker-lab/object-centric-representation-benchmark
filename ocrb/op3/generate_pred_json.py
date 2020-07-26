import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

from ocrb.eval.eval_utils import binarize_masks, rle_encode
from exp_variants.variants import *
import pytorch_util as ptu
import op3_modules.op3_model as op3_model
from op3_modules.op3_trainer import TrainingScheduler
from ocrb.data.dataloader import build_testloader

parser = argparse.ArgumentParser()
parser.add_argument('-va', '--variant', type=str, default='vmds', choices=['vmds', 'vor', 'spmot'])
parser.add_argument('--ckpt_file', help='Path to ckpt file.', type=str)
parser.add_argument('--out_path', help='Path to output file.', type=str)


def generate_annotation_file(model, scheduler, test_loader, n_steps=10, n_slots=6, path=None, device=None):
    ''' Generate json file containing mask and object id predictions per frame for each video in testset.
    '''
    pred_list = []
    id_counter = 0
    schedule = scheduler.get_schedule(0, is_train=False)
    pred_indices = [i for i, x in enumerate(schedule) if x == 1]
    rec_indices = [i-1 for i in pred_indices]
    rec_indices.append(-1)
    loss_schedule = scheduler.get_loss_schedule(schedule)
    for i, data in tqdm(enumerate(test_loader)):
        bs = data.size(0)
        #perform inference
        _, results, _, _, _, _, _, _ = model.forward(data.float().to(device), None, 
                                                     initial_hidden_state=None, schedule=schedule,
                                        loss_schedule=loss_schedule)
        soft_masks = results.cpu()
        for b in range(bs):
            video = []
            obj_ids = np.arange(n_slots) + id_counter
            for t in range(n_steps):
                binarized_masks = binarize_masks(soft_masks[b,rec_indices[t]].squeeze())
                binarized_masks = np.array(binarized_masks).astype(np.uint8)

                frame = {}
                masks = []
                ids = []
                for j in range(n_slots):
                    # ignore slots with empty masks
                    if binarized_masks[j].sum() == 0.:
                        continue
                    else:
                        masks.append(rle_encode(binarized_masks[j]))
                        ids.append(int(obj_ids[j]))
                frame['masks'] = masks
                frame['ids'] = ids
                video.append(frame)
            
            pred_list.append(video)
            id_counter += n_slots  

    with open(path, 'w') as outfile:
        json.dump(pred_list, outfile)
     
    
def main(args):
    # load config
    if args.variant == 'vmds':
        variant = multidsprites_videos_variant
    elif args.variant == 'spmot':
        variant = mot_sprite_variant
    elif args.variant == 'vor':
        variant = object_room_variant
    
    # load data
    print('Loading dataset: {}'.format(args.variant))
    test_loader = build_testloader(batch_size=variant['training_args']['batch_size'], num_workers=variant['num_workers'], n_steps=variant['n_steps'], dataset_class=args.variant, path=variant['path'])

    # build model 
    if torch.cuda.is_available():
        ptu.set_gpu_mode(True)
    model = op3_model.create_model_v2(variant['op3_args'], variant['op3_args']['det_repsize'], variant['op3_args']['sto_repsize'], action_dim=0)
    # load ckpt
    state_dict = torch.load(args.ckpt_file, map_location=lambda storage, loc: storage)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if 'module.' in k:
            name = k[7:]  # remove 'module.' of dataparallel
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval_mode = True
    if variant['dataparallel']:
        model = torch.nn.DataParallel(model)
    model.to(ptu.device)
    variant['schedule_args']['T'] = variant['n_steps']
    scheduler = TrainingScheduler(**variant["schedule_args"], max_T = variant['n_steps'])
    
    generate_annotation_file(model, 
                             scheduler,
                             test_loader,
                             n_steps=variant['n_steps'], 
                             n_slots=variant['op3_args']['K'],
                             path=args.out_path,
                             device=ptu.device)
        

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
