import json
import torch
import argparse
import numpy as np
from tqdm import tqdm

from ocrb.eval.eval_utils import binarize_masks, rle_encode
from ocrb.vimon.networks.model import build_vimon
from ocrb.data.dataloader import build_testloader

parser = argparse.ArgumentParser()
parser.add_argument('--config', help='Path to config file.', type=str)
parser.add_argument('--ckpt_file', help='Path to ckpt file.', type=str)
parser.add_argument('--out_path', help='Path to output file.', type=str)


def generate_annotation_file(model, test_loader, n_steps=10, n_slots=6, key='curr_masks', path=None, device=None):
    ''' Generate json file containing mask and object id predictions per frame for each video in testset.
    '''
    pred_list = []
    id_counter = 0
    for i, data in tqdm(enumerate(test_loader)):
        bs = data.size(0)
        #perform inference
        with torch.no_grad():
            results, _ = model(data.float().to(device))
        soft_masks = results[key].cpu()
        
        for b in range(bs):
            video = []
            obj_ids = np.arange(n_slots) + id_counter
            for t in range(n_steps):
                binarized_masks = binarize_masks(soft_masks[b,t])
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load config
    config = json.load(open(args.config))
    
    # load data
    print('Loading {}'.format(config['data']['name']))
    test_loader = build_testloader(batch_size=config['data']['batch_size'], num_workers=config['data']['num_workers'], n_steps=config['data']['n_steps'], dataset_class=config['data']['name'], path=config['data']['path'])

    # build model 
    model = build_vimon(config, inference=True).to(device)
    # load ckpt
    ckpt = torch.load(args.ckpt_file, map_location=lambda storage, loc: storage)
    model.load_state_dict(ckpt)
    
    generate_annotation_file(model, 
                             test_loader,
                             n_steps=config['data']['n_steps'], 
                             n_slots=config['model']['n_slots'],
                             path=args.out_path,
                             device=device)
        

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
