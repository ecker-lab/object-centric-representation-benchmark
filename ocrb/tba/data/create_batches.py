import numpy as np
np.random.seed(1)
import torch
import argparse

FRAMES = 10

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', help='Batch size.', type=int, default=1)
parser.add_argument('--dataset', help='Dataset.', type=str, default='vmds')
parser.add_argument('--mode', help='Train, validation or test set.', type=str, default='train')


def main(args):
    assert args.mode in ['train', 'val', 'test']
    assert args.dataset in ['vmds', 'vor', 'spmot']

    train = True if args.mode == 'train' else False
    batch_size = args.batch_size
    data = np.load('ocrb/data/datasets/{}/{}_{}.npy'.format(args.dataset, args.dataset, args.mode))
    
    if train:
        np.random.shuffle(data)

    SIZE = len(data) // batch_size
    _, _, CHANNELS, WIDTH, HEIGHT = data.shape

    for batch_number in range(SIZE):
        batch = torch.Tensor(data[batch_number*batch_size:(batch_number+1)*batch_size, :FRAMES]).view(-1, CHANNELS, WIDTH, HEIGHT)
        batch = torch.nn.functional.interpolate(batch, scale_factor=2).view(batch_size, FRAMES, CHANNELS, WIDTH*2, HEIGHT*2)
        if args.mode == 'train' or args.mode == 'val':
            filename = 'ocrb/tba/data/{}/pt/input/{}_'.format(args.dataset, args.mode) + str(batch_number) + '.pt'
        else:
            filename = 'ocrb/tba/data/{}/pt/metric/test_'.format(args.dataset) + str(batch_number) + '.pt'
        print(filename, batch.shape)
        torch.save(batch.type(torch.uint8), filename)

    
if __name__ == '__main__':    
    args = parser.parse_args()
    main(args)