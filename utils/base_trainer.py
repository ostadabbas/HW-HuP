from __future__ import division
import sys
import time

import torch
from tqdm import tqdm
tqdm.monitor_interval = 0
from torch.utils.tensorboard import SummaryWriter
from utils import CheckpointDataLoader, CheckpointSaver
import time

class BaseTrainer(object):
    """Base class for Trainer objects.
    Takes care of checkpointing/logging/resuming training.
    """
    def __init__(self, options, phase='train'):
        self.options = options
        self.endtime = time.time() + self.options.time_to_run
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # override this function to define your model, optimizers etc.
        self.init_fn(phase=phase)
        self.saver = CheckpointSaver(save_dir=options.checkpoint_dir)
        self.summary_writer = SummaryWriter(self.options.summary_dir, flush_secs=60)
        # time.sleep(5)

        self.checkpoint = None
        if self.options.resume and self.saver.exists_checkpoint():
            self.checkpoint = self.saver.load_checkpoint(self.models_dict, self.optimizers_dict, checkpoint_file=self.options.checkpoint)   # latest check point

        if self.checkpoint is None: # defult is none  excep tresume
            self.epoch_count = 0
            self.step_count = 0
        else:
            self.epoch_count = self.checkpoint['epoch']
            self.step_count = self.checkpoint['total_step_count']
        if options.train_ds == 'h36m_2d':
            self.if_rend_ptc = False        # mpii no ptc
        if options.train_ds == 'infant':
            self.if_rend_ptc = False        # SyRIP no ptc s


    def load_pretrained(self, checkpoint_file=None):
        """Load a pretrained checkpoint.
        This is different from resuming training using --resume.
        """
        if checkpoint_file is not None:
            checkpoint = torch.load(checkpoint_file)
            for model in self.models_dict:
                if model in checkpoint:
                    self.models_dict[model].load_state_dict(checkpoint[model], strict=False)
                    print('Checkpoint loaded')

    def train(self):
        """Training process."""
        # Run training for num_epochs epochs
        epoch_dp = self.options.epoch_dp
        epoch_cam = self.options.epoch_cam
        trainIter = self.options.trainIter
        opt_wt_pose_dRt = self.options.opt_wt_pose_dRt

        end_epoch = min(self.epoch_count+ self.options.epoch_step, self.options.num_epochs)

        for epoch in tqdm(range(self.epoch_count, self.options.num_epochs), total=self.options.num_epochs, initial=self.epoch_count):
            # Create new DataLoader every epoch and (possibly) resume from an arbitrary step inside an epoch
            train_data_loader = CheckpointDataLoader(self.train_ds, checkpoint=self.checkpoint,
                                                     batch_size=self.options.batch_size,
                                                     num_workers=self.options.num_workers,
                                                     pin_memory=self.options.pin_memory,
                                                     shuffle=self.options.shuffle_train)
            if epoch >= end_epoch:      # contol the steps
                break

            # update the trainer state
            if epoch_dp >= 0 and epoch >= epoch_dp:  # according to epoch , update the current training state
                flg_dp = 1
            else:
                flg_dp = 0
                # self.update_state(flg_dp=1)  # 1 for 3d dp state
            if epoch_cam >= 0 and epoch >= epoch_cam:  # according to epoch , update the current training state
                flg_cam = 0
            else:
                flg_cam = 1
            self.update_state(flg_dp=flg_dp, flg_cam=flg_cam)

            # update the opt ratio
            if epoch>0 and  opt_wt_pose_dRt > 0:
                self.options.opt_wt_pose = self.options.opt_wt_pose *(opt_wt_pose_dRt**epoch)   # wh equals to exp total number
                # print('current opt_wt', self.options.opt_wt_pose)    # debug, working

            # Iterate over all batches in an epoch
            for step, batch in enumerate(tqdm(train_data_loader, desc='Epoch '+str(epoch),
                                              total=len(self.train_ds) // self.options.batch_size,
                                              initial=train_data_loader.checkpoint_batch_idx),
                                         train_data_loader.checkpoint_batch_idx): # enumerate(value, start)
                if trainIter>0 and step>= trainIter:        # stop the train iteration
                    break

                if time.time() < self.endtime:
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
                    out = self.train_step(batch)    # output loss
                    if out[0] is None:
                        continue
                    self.step_count += 1
                    # Tensorboard logging every summary_steps steps
                    if self.step_count % self.options.summary_steps == 0 :   # every 100 steps , save 1 summary, certain step write the table
                        self.train_summaries(batch, *out, if_rend_ptc=self.if_rend_ptc)   # save summary, -> batch, output, losses,
                        # print('WIRTE scalar {} at step {}'.format(step % 10, step))
                        # self.summary_writer.add_scalar('test', step % 10, step)   # just write the random scalar
                    # Save checkpoint every checkpoint_steps steps
                    ckpt_step = self.options.checkpoint_steps
                    if ckpt_step > 0 and self.step_count % self.options.checkpoint_steps == 0:
                        self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch, step+1, self.options.batch_size, train_data_loader.sampler.dataset_perm, self.step_count)
                        tqdm.write('Checkpoint saved')

                    # Run validation every test_steps steps
                    if self.step_count % self.options.test_steps == 0:
                        self.test()
                else:
                    tqdm.write('Timeout reached')
                    self.finalize()
                    self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch, step, self.options.batch_size, train_data_loader.sampler.dataset_perm, self.step_count)       # logs/exp/checkpoints/date.pt
                    tqdm.write('Checkpoint saved')
                    sys.exit(0)

            # load a checkpoint only on startup, for the next epochs
            # just iterate over the dataset as usual
            self.checkpoint=None
            # save checkpoint after each epoch
            # if (epoch+1) % 10 == 0:     # every 10 epoch save it
            if (epoch+1) % self.options.checkpoint_epoch_steps == 0:     # every 10 epoch save it
                # self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch+1, 0, self.step_count)
                self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch+1, 0, self.options.batch_size, None, self.step_count)
        return

    # The following methods (with the possible exception of test) have to be implemented in the derived classes
    def init_fn(self):
        raise NotImplementedError('You need to provide an _init_fn method')

    def train_step(self, input_batch):
        raise NotImplementedError('You need to provide a _train_step method')

    def train_summaries(self, input_batch):
        raise NotImplementedError('You need to provide a _train_summaries method')

    def test(self):
        pass

    def update_state(self, code=0): # update the state according to the given code
        pass
