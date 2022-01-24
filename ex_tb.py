'''
test the tensorboard usage
'''
import tensorboard
import os
import os.path as osp
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

tb_fd = 'tbEx_fd'
if not osp.exists(tb_fd):
	os.makedirs(tb_fd)

## summary writer
# summary_wr = SummaryWriter(tb_fd)
# for i in tqdm(range(10), 'writing pseudo data') :
# 	summary_wr.add_scalar('haha', i*2, i)
# 	summary_wr.add_scalar('mama', i*3, i)


writer = SummaryWriter()
for n_iter in tqdm(range(100),'writing summaries'):
	writer.add_scalar('Loss/train', np.random.random(), n_iter)
	writer.add_scalar('Loss/test', np.random.random(), n_iter)
	writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
	writer.add_scalar('Accuracy/test', np.random.random(), n_iter)

writer.flush()
# writer.close()      # has to close , it works here.

