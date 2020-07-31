import numpy as np
np.random.seed(1)
import torch

BATCH_SIZE = 1
FRAMES = 10

# train_data = np.load('multidsprites_videos_train_black_bg2.npy')
# np.random.shuffle(train_data)
val_data = np.load('multidsprites_videos_test_sym_black_bg.npy')

# TRAIN_SIZE = len(train_data)//BATCH_SIZE
VAL_SIZE = len(val_data)//BATCH_SIZE
_, _, CHANNELS, WIDTH, HEIGHT = val_data.shape

# for batch_number in range(TRAIN_SIZE):
#     batch = torch.Tensor(train_data[batch_number*BATCH_SIZE:(batch_number+1)*BATCH_SIZE, :FRAMES]).view(-1, CHANNELS, WIDTH, HEIGHT)
#     batch = torch.nn.functional.interpolate(batch, scale_factor=2).view(BATCH_SIZE, FRAMES, CHANNELS, WIDTH*2, HEIGHT*2)
#     filename = 'input/train_' + str(batch_number) + '.pt'
#     print(filename, batch.shape)
#     torch.save(batch.type(torch.uint8), filename)

for batch_number in range(VAL_SIZE):
    batch = torch.Tensor(val_data[batch_number*BATCH_SIZE:(batch_number+1)*BATCH_SIZE, :FRAMES]).view(-1, CHANNELS, WIDTH, HEIGHT)
    batch = torch.nn.functional.interpolate(batch, scale_factor=2).view(BATCH_SIZE, FRAMES, CHANNELS, WIDTH*2, HEIGHT*2)
    # filename = 'input/test_' + str(batch_number) + '.pt'
    filename = 'metric/sym/test_' + str(batch_number) + '.pt'
    print(filename, batch.shape)
    torch.save(batch.type(torch.uint8), filename)
