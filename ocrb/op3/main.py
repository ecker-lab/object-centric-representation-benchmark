import torch
import os
import json
import argparse
import op3_modules.op3_model as op3_model
from op3_modules.op3_trainer import TrainingScheduler, OP3Trainer
from ocrb.data.dataloader import build_dataloader
from exp_variants.variants import *
from pytorch_util import set_gpu_mode
from core import logger

parser = argparse.ArgumentParser()
parser.add_argument('-va', '--variant', type=str, default='vmds', choices=['vmds', 'vor', 'spmot'])


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
    train_loader, val_loader = build_dataloader(batch_size=variant['training_args']['batch_size'], num_workers=variant['num_workers'], n_steps=variant['n_steps'], dataset_class=args.variant, path=variant['path'], T=variant['schedule_args']['T'])
    os.makedirs(variant['ckpt_dir'], exist_ok=True)
    logger.set_snapshot_dir(variant['ckpt_dir'])
    # build model 
    if torch.cuda.is_available():
        set_gpu_mode(True)
    model = op3_model.create_model_v2(variant['op3_args'], variant['op3_args']['det_repsize'], variant['op3_args']['sto_repsize'], action_dim=0)
    if variant['dataparallel']:
        model = torch.nn.DataParallel(model)
    scheduler = TrainingScheduler(**variant["schedule_args"], max_T = variant['n_steps'])
    trainer = OP3Trainer(train_loader, val_loader, model, scheduler, **variant["training_args"])

    print('Start training.')        
    for epoch in range(variant['num_epochs']):
        should_save_imgs = (epoch % variant['save_period'] == 0)
        train_stats = trainer.train_epoch(epoch)
        test_stats = trainer.test_epoch(epoch, train=False, batches=1, save_reconstruction=should_save_imgs)
        trainer.test_epoch(epoch, train=True, batches=1, save_reconstruction=should_save_imgs)
        for k, v in {**train_stats, **test_stats}.items():
            logger.record_tabular(k, v)
        logger.dump_tabular()
        trainer.save_model()
    print('Done.')


if __name__ == '__main__':    
    args = parser.parse_args()
    main(args)
