import json
import argparse
from train import Trainer
from networks.model import build_vimon
from ocrb.data.dataloader import build_dataloader

parser = argparse.ArgumentParser()
parser.add_argument('--config', help='Path to config file.', type=str, default='object_centric/vimon/config.json')


def main(args):
    # load config
    config = json.load(open(args.config))
    
    # load data
    print('Loading dataset: {}'.format(config['data']['name']))
    train_loader, val_loader = build_dataloader(batch_size=config['data']['batch_size'], num_workers=config['data']['num_workers'], n_steps=config['data']['n_steps'], dataset_class=config['data']['name'], path=config['data']['path'])

    # build model 
    model = build_vimon(config)
    trainer = Trainer(config, model, [train_loader, val_loader])

    print('Start training.')
    trainer.train()
    print('Done.')


if __name__ == '__main__':    
    args = parser.parse_args()
    main(args)
