from __future__ import print_function

import sys

sys.path.insert(0, '.')

import torch
from torch.autograd import Variable
import torch.optim as optim
from torch.nn.parallel import DataParallel

import time
import os.path as osp
from tensorboardX import SummaryWriter
import numpy as np
import argparse

from tri_loss.dataset import create_dataset
from tri_loss.model.Model import Model
from tri_loss.model.TripletLoss import TripletLoss
from tri_loss.model.loss import global_loss

from tri_loss.utils.utils import time_str
from tri_loss.utils.utils import str2bool
from tri_loss.utils.utils import tight_float_str as tfs
from tri_loss.utils.utils import may_set_mode
from tri_loss.utils.utils import load_state_dict
from tri_loss.utils.utils import load_ckpt
from tri_loss.utils.utils import save_ckpt
from tri_loss.utils.utils import set_devices
from tri_loss.utils.utils import AverageMeter
from tri_loss.utils.utils import to_scalar
from tri_loss.utils.utils import ReDirectSTD
from tri_loss.utils.utils import set_seed
from tri_loss.utils.utils import adjust_lr_exp
from tri_loss.utils.utils import adjust_lr_staircase

from tensorboardX import SummaryWriter
writer = SummaryWriter()

class Config(object):
  def __init__(self):

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--sys_device_ids', type=eval, default=(0,))
    parser.add_argument('-r', '--run', type=int, default=1)
    parser.add_argument('--set_seed', type=str2bool, default=False)
    parser.add_argument('--dataset', type=str, default='market1501',
                        choices=['market1501', 'cuhk03', 'duke', 'combined'])
    parser.add_argument('--trainset_part', type=str, default='trainval',
                        choices=['trainval', 'train'])

    parser.add_argument('--resize_h_w', type=eval, default=(384, 128))     #(256, 128)  (384, 128)  (480, 160)
    # These several only for training set
    parser.add_argument('--crop_prob', type=float, default=0)
    parser.add_argument('--crop_ratio', type=float, default=1)
    parser.add_argument('--mirror', type=str2bool, default=True)
    parser.add_argument('--ids_per_batch', type=int, default=14)   #32  16
    parser.add_argument('--ims_per_id', type=int, default=6)

    parser.add_argument('--log_to_file', type=str2bool, default=True)
    parser.add_argument('--steps_per_log', type=int, default=20)
    parser.add_argument('--epochs_per_val', type=int, default=1e10)

    parser.add_argument('--last_conv_stride', type=int, default=1,
                        choices=[1, 2])
    parser.add_argument('--normalize_feature', type=str2bool, default=False)
    parser.add_argument('--margin', type=float, default=0.3)                 #0.3

    parser.add_argument('--only_test', type=str2bool, default=False)
    parser.add_argument('--resume', type=str2bool, default=False) #False
    parser.add_argument('--exp_dir', type=str, default='')
    parser.add_argument('--model_weight_file', type=str, default='')

    parser.add_argument('--base_lr', type=float, default=2e-4)   #2e-4
    parser.add_argument('--lr_decay_type', type=str, default='exp',    #'exp'
                        choices=['exp', 'staircase'])
    parser.add_argument('--exp_decay_at_epoch', type=int, default=260)      #151  210
    parser.add_argument('--staircase_decay_at_epochs',
                        type=eval, default=(301, 401, 451, 501, 551, 601, 651, 701, 751))    #(101, 201,)
    parser.add_argument('--staircase_decay_multiply_factor',
                        type=float, default=0.3)       #0.1
    parser.add_argument('--total_epochs', type=int, default=600)      #300

    args = parser.parse_args()

    # gpu ids
    self.sys_device_ids = args.sys_device_ids

    # If you want to make your results exactly reproducible, you have
    # to fix a random seed.
    if args.set_seed:
      self.seed = 1
    else:
      self.seed = None

    # The experiments can be run for several times and performances be averaged.
    # `run` starts from `1`, not `0`.
    self.run = args.run

    ###########
    # Dataset #
    ###########

    # If you want to make your results exactly reproducible, you have
    # to also set num of threads to 1 during training.
    if self.seed is not None:
      self.prefetch_threads = 1
    else:
      self.prefetch_threads = 2

    self.dataset = args.dataset
    self.trainset_part = args.trainset_part

    # Image Processing

    # Just for training set
    self.crop_prob = args.crop_prob
    self.crop_ratio = args.crop_ratio
    self.resize_h_w = args.resize_h_w

    # Whether to scale by 1/255
    self.scale_im = True
    self.im_mean = [0.486, 0.459, 0.408]
    self.im_std = [0.229, 0.224, 0.225]

    self.train_mirror_type = 'random' if args.mirror else None

    self.ids_per_batch = args.ids_per_batch
    self.ims_per_id = args.ims_per_id
    self.train_final_batch = False  #False
    self.train_shuffle = True

    self.test_batch_size = 32
    self.test_final_batch = True
    self.test_mirror_type = None
    self.test_shuffle = False

    dataset_kwargs = dict(
      name=self.dataset,
      resize_h_w=self.resize_h_w,
      scale=self.scale_im,
      im_mean=self.im_mean,
      im_std=self.im_std,
      batch_dims='NCHW',
      num_prefetch_threads=self.prefetch_threads)

    prng = np.random
    if self.seed is not None:
      prng = np.random.RandomState(self.seed)
    self.train_set_kwargs = dict(
      part=self.trainset_part,
      ids_per_batch=self.ids_per_batch,
      ims_per_id=self.ims_per_id,
      final_batch=self.train_final_batch,
      shuffle=self.train_shuffle,
      crop_prob=self.crop_prob,
      crop_ratio=self.crop_ratio,
      mirror_type=self.train_mirror_type,
      prng=prng)
    self.train_set_kwargs.update(dataset_kwargs)

    prng = np.random
    if self.seed is not None:
      prng = np.random.RandomState(self.seed)
    self.val_set_kwargs = dict(
      part='val',
      batch_size=self.test_batch_size,
      final_batch=self.test_final_batch,
      shuffle=self.test_shuffle,
      mirror_type=self.test_mirror_type,
      prng=prng)
    self.val_set_kwargs.update(dataset_kwargs)

    prng = np.random
    if self.seed is not None:
      prng = np.random.RandomState(self.seed)
    self.test_set_kwargs = dict(
      part='test',
      batch_size=self.test_batch_size,
      final_batch=self.test_final_batch,
      shuffle=self.test_shuffle,
      mirror_type=self.test_mirror_type,
      prng=prng)
    self.test_set_kwargs.update(dataset_kwargs)

    ###############
    # ReID Model  #
    ###############

    # The last block of ResNet has stride 2. We can set the stride to 1 so that
    # the spatial resolution before global pooling is doubled.
    self.last_conv_stride = args.last_conv_stride

    # Whether to normalize feature to unit length along the Channel dimension,
    # before computing distance
    self.normalize_feature = args.normalize_feature

    # Margin of triplet loss
    self.margin = args.margin

    #############
    # Training  #
    #############

    self.weight_decay = 0.0005

    # Initial learning rate
    self.base_lr = args.base_lr
    self.lr_decay_type = args.lr_decay_type
    self.exp_decay_at_epoch = args.exp_decay_at_epoch
    self.staircase_decay_at_epochs = args.staircase_decay_at_epochs
    self.staircase_decay_multiply_factor = args.staircase_decay_multiply_factor
    # Number of epochs to train
    self.total_epochs = args.total_epochs

    # How often (in epochs) to test on val set.
    self.epochs_per_val = args.epochs_per_val

    # How often (in batches) to log. If only need to log the average
    # information for each epoch, set this to a large value, e.g. 1e10.
    self.steps_per_log = args.steps_per_log

    # Only test and without training.
    self.only_test = args.only_test

    self.resume = args.resume

    #######
    # Log #
    #######

    # If True,
    # 1) stdout and stderr will be redirected to file,
    # 2) training loss etc will be written to tensorboard,
    # 3) checkpoint will be saved
    self.log_to_file = args.log_to_file

    # The root dir of logs.
    if args.exp_dir == '':
      self.exp_dir = osp.join(
        'exp/train',
        '{}'.format(self.dataset),
        #
        'lcs_{}_'.format(self.last_conv_stride) +
        ('nf_' if self.normalize_feature else 'not_nf_') +
        'margin_{}_'.format(tfs(self.margin)) +
        'lr_{}_'.format(tfs(self.base_lr)) +
        '{}_'.format(self.lr_decay_type) +
        ('decay_at_{}_'.format(self.exp_decay_at_epoch)
         if self.lr_decay_type == 'exp'
         else 'decay_at_{}_factor_{}_'.format(
          '_'.join([str(e) for e in args.staircase_decay_at_epochs]),
          tfs(self.staircase_decay_multiply_factor))) +
        'total_{}'.format(self.total_epochs),
        #
        'run{}'.format(self.run),
      )
    else:
      self.exp_dir = args.exp_dir

    self.stdout_file = osp.join(
      self.exp_dir, 'stdout_{}.txt'.format(time_str()))
    self.stderr_file = osp.join(
      self.exp_dir, 'stderr_{}.txt'.format(time_str()))

    # Saving model weights and optimizer states, for resuming.

    self.ckpt_file = osp.join(self.exp_dir, 'ckpt.pth')

    # Just for loading a pretrained model; no optimizer states is needed.
    self.model_weight_file = args.model_weight_file


class ExtractFeature(object):
  """A function to be called in the val/test set, to extract features.
  Args:
    TVT: A callable to transfer images to specific device.
  """

  def __init__(self, model, TVT):
    self.model = model
    self.TVT = TVT

  def __call__(self, ims):
    old_train_eval_model = self.model.training
    # Set eval mode.
    # Force all BN layers to use global mean and variance, also disable
    # dropout.
    self.model.eval()
    ims = Variable(self.TVT(torch.from_numpy(ims).float()))
    '''
    feat, c_class, yy1, yy2, yy3, yy4, yy5, yy6, yy7, yy8, yy9, yy10, xx, mm1, mm2, mm3, mm4, mm5, mm6, mm7, mm8, mm9, mm10,yyy = self.model(ims)
    feat = feat.data.cpu().numpy()
    yy1 = yy1.data.cpu().numpy()
    yy2 = yy2.data.cpu().numpy()
    yy3 = yy3.data.cpu().numpy()
    yy4 = yy4.data.cpu().numpy()
    yy5 = yy5.data.cpu().numpy()
    yy6 = yy6.data.cpu().numpy()
    yy7 = yy7.data.cpu().numpy()
    yy8 = yy8.data.cpu().numpy()
    yy9 = yy9.data.cpu().numpy()
    yy10 = yy10.data.cpu().numpy()
    mm1 = mm1.data.cpu().numpy()
    mm2 = mm2.data.cpu().numpy()
    mm3 = mm3.data.cpu().numpy()
    mm4 = mm4.data.cpu().numpy()
    mm5 = mm5.data.cpu().numpy()
    mm6 = mm6.data.cpu().numpy()
    mm7 = mm7.data.cpu().numpy()
    mm8 = mm8.data.cpu().numpy()
    mm9 = mm9.data.cpu().numpy()
    mm10 = mm10.data.cpu().numpy()
    xx = xx.data.cpu().numpy()
    yyy = yyy.data.cpu().numpy()
    '''

    #feature_maps = []

    #def hook(module, input, feature_map):
    #  feature_maps.append(feature_map)

    #hk = self.mask1[1].register_forward_hook(hook)
    #hk = self.model.Model.mask1[1].register_forward_hook(hook)
    label_one = torch.ones(ims.size(0),1502)

    fea, feat0, c_cla0, feat1, c_cla1,  feat2, c_cla2, feat3, c_cla3, feat4, c_cla4 = self.model(ims,label_one)

    #self.model.train(old_train_eval_model)
    #hk.remove()

    fea = fea.data.cpu().numpy()
    #feat = feat.data.cpu().numpy()
    #ou_glo = ou_glo.data.cpu().numpy()
    #ou_all = ou_all.data.cpu().numpy()
    #ff2 = ff2.data.cpu().numpy()
    #ff1 = ff1.data.cpu().numpy()
    #ffff = ffff.data.cpu().numpy()

    # Restore the model to its old train/eval mode.
    self.model.train(old_train_eval_model)
    #return feat, yy1, yy2, yy3, yy4, yy5, yy6, yy7, yy8, yy9, yy10, xx, mm1, mm2, mm3, mm4, mm5, mm6, mm7, mm8, mm9, mm10,yyy
    #return feat
    return fea


def main():
  cfg = Config()
  lossss = torch.nn.CrossEntropyLoss()
  # Redirect logs to both console and file.
  if cfg.log_to_file:
    ReDirectSTD(cfg.stdout_file, 'stdout', False)
    ReDirectSTD(cfg.stderr_file, 'stderr', False)

  # Lazily create SummaryWriter
  writer = None

  TVT, TMO = set_devices(cfg.sys_device_ids)

  if cfg.seed is not None:
    set_seed(cfg.seed)

  # Dump the configurations to log.
  import pprint
  print('-' * 60)
  print('cfg.__dict__')
  pprint.pprint(cfg.__dict__)
  print('-' * 60)

  ###########
  # Dataset #
  ###########

  if not cfg.only_test:
    train_set = create_dataset(**cfg.train_set_kwargs)
    #print (train_set.im_names)
    # The combined dataset does not provide val set currently.
    val_set = None if cfg.dataset == 'combined' else create_dataset(**cfg.val_set_kwargs)

  test_sets = []
  test_set_names = []
  if cfg.dataset == 'combined':
    for name in ['market1501', 'cuhk03', 'duke']:
      cfg.test_set_kwargs['name'] = name
      test_sets.append(create_dataset(**cfg.test_set_kwargs))
      test_set_names.append(name)
  else:
    test_sets.append(create_dataset(**cfg.test_set_kwargs))
    test_set_names.append(cfg.dataset)

  ###########
  # Models  #
  ###########

  model = Model(last_conv_stride=cfg.last_conv_stride)

  #model_ini = Model(last_conv_stride=cfg.last_conv_stride)
  #model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_ini)

  # Model wrapper
  model_w = DataParallel(model)

  print (model)
  #############################
  # Criteria and Optimizers   #
  #############################

  tri_loss = TripletLoss(margin=cfg.margin)

  optimizer = optim.Adam(model.parameters(),
                         lr=cfg.base_lr,
                         weight_decay=cfg.weight_decay)

  # Bind them together just to save some codes in the following usage.
  modules_optims = [model, optimizer]

  ################################
  # May Resume Models and Optims #
  ################################

  TMO(modules_optims)

  if cfg.resume:
    #ckptt = cfg.ckpt_file + str(149)
    #resume_ep, scores = load_ckpt(modules_optims, ckptt)
    resume_ep, scores = load_ckpt(modules_optims, cfg.ckpt_file)

  # May Transfer Models and Optims to Specified Device. Transferring optimizer
  # is to cope with the case when you load the checkpoint to a new device.
  #TMO(modules_optims)

  ########
  # Test #
  ########

  def test(load_model_weight=False):
    if load_model_weight:
      if cfg.model_weight_file != '':
        map_location = (lambda storage, loc: storage)
        sd = torch.load(cfg.model_weight_file, map_location=map_location)
        load_state_dict(model, sd)
        print('Loaded model weights from {}'.format(cfg.model_weight_file))
      else:
        #ckptt = cfg.ckpt_file + str(299)
        #load_ckpt(modules_optims, ckptt)
        load_ckpt(modules_optims, cfg.ckpt_file)

    for test_set, name in zip(test_sets, test_set_names):
      test_set.set_feat_func(ExtractFeature(model_w, TVT))
      print('\n=========> Test on dataset: {} <=========\n'.format(name))
      test_set.eval(
        normalize_feat=cfg.normalize_feature,
        verbose=True)

  def validate():
    if val_set.extract_feat_func is None:
      val_set.set_feat_func(ExtractFeature(model_w, TVT))
    print('\n=========> Test on validation set <=========\n')
    mAP, cmc_scores, _, _ = val_set.eval(
      normalize_feat=cfg.normalize_feature,
      to_re_rank=False,
      verbose=False)
    print()
    return mAP, cmc_scores[0]

  if cfg.only_test:
    test(load_model_weight=True)
    return

  ############
  # Training #
  ############

  start_ep = resume_ep if cfg.resume else 0
  for ep in range(start_ep, 550):    #cfg.total_epochs   550  600
    # Adjust Learning Rate
    if cfg.lr_decay_type == 'exp':
      adjust_lr_exp(
        optimizer,
        cfg.base_lr,
        ep + 1,
        550,    #cfg.total_epochs
        cfg.exp_decay_at_epoch)
    else:
      adjust_lr_staircase(
        optimizer,
        cfg.base_lr,
        ep + 1,
        cfg.staircase_decay_at_epochs,
        cfg.staircase_decay_multiply_factor)

    may_set_mode(modules_optims, 'train')

    # For recording precision, satisfying margin, etc
    prec_meter = AverageMeter()
    sm_meter = AverageMeter()
    dist_ap_meter = AverageMeter()
    dist_an_meter = AverageMeter()
    loss_meter = AverageMeter()

    ep_st = time.time()
    step = 0
    epoch_done = False
    while not epoch_done:

      step += 1
      step_st = time.time()

      ims, im_names, labels, mirrored, epoch_done = train_set.next_batch()

      ims_var = Variable(TVT(torch.from_numpy(ims).float()))         #[128, 3, 256, 128]
      #print (ims_var.size())
      labels_t = TVT(torch.from_numpy(labels).long())                #[128]

      ## one-hot ##
      lab = Variable(labels_t)
      b = torch.LongTensor(labels)
      c = b.view(84,1)   #128 64

      y_onehot = torch.FloatTensor(84, 1502)
      # In your for loop
      y_onehot.zero_()
      y_onehot.scatter_(1, c, 1)

      #
      y_onehot1 = y_onehot
      #

      y_onehot = y_onehot*0.7     #0.8
      po = 0.3 / 1501
      piil = torch.zeros(84,1502)
      pik = piil +  po
      labab = torch.where(y_onehot>pik,y_onehot,pik)
      labab = labab.cuda()
      #lossss = torch.nn.CrossEntropyLoss()
      fea, feat, c_cla, feat1, c_cla1, feat2, c_cla2, feat3, c_cla3, feat4, c_cla4 = model_w(ims_var,y_onehot1)     #[128,2048]
      #feat, c_class = model_w(ims_var)  # [128,2048]
      '''
      ## cross loss ##
      cros = c_class * labab
      loss1 = -cros.sum(dim=1)
      loss1 = loss1.mean()

      #loss1 = lossss(c_class, lab)

      loss2, p_inds, n_inds, dist_ap, dist_an, dist_mat = global_loss(
        tri_loss, feat, labels_t,
        normalize_feature=cfg.normalize_feature)
      '''
      #cros1 = c_cla1 * labab
      #loss1 = -cros1.sum(dim=1)
      #loss1 = loss1.mean()

      #cros2 = c_cla2 * labab
      #loss2 = -cros2.sum(dim=1)
      #loss2 = loss2.mean()


      cros6 = c_cla * labab
      loss13 = -cros6.sum(dim=1)
      loss13 = loss13.mean()

      loss14, p_inds, n_inds, dist_ap, dist_an, dist_mat = global_loss(
        tri_loss, feat, labels_t,
        normalize_feature=cfg.normalize_feature)

      loss7, p_inds, n_inds, dist_ap, dist_an, dist_mat = global_loss(
        tri_loss, feat1, labels_t,
        normalize_feature=cfg.normalize_feature)

      cros3 = c_cla1 * labab
      loss8 = -cros3.sum(dim=1)
      loss8 = loss8.mean()

      loss9, p_inds, n_inds, dist_ap, dist_an, dist_mat = global_loss(
        tri_loss, feat2, labels_t,
        normalize_feature=cfg.normalize_feature)

      cros4 = c_cla2 * labab
      loss10 = -cros4.sum(dim=1)
      loss10 = loss10.mean()

      loss11, p_inds, n_inds, dist_ap, dist_an, dist_mat = global_loss(
        tri_loss, feat3, labels_t,
        normalize_feature=cfg.normalize_feature)

      cros5 = c_cla3 * labab
      loss12 = -cros5.sum(dim=1)
      loss12 = loss12.mean()


      cros1 = c_cla4 * labab
      loss3 = -cros1.sum(dim=1)
      loss3 = loss3.mean()

      loss4, p_inds, n_inds, dist_ap, dist_an, dist_mat = global_loss(
        tri_loss, feat4, labels_t,
        normalize_feature=cfg.normalize_feature)
      '''
      cros2 = c_class2 * labab
      loss5 = -cros2.sum(dim=1)
      loss5 = loss5.mean()

      loss6, p_inds, n_inds, dist_ap, dist_an, dist_mat = global_loss(
        tri_loss, feat2, labels_t,
        normalize_feature=cfg.normalize_feature)
      '''
      '''
      cros2 = cc2 * labab
      loss5 = -cros2.sum(dim=1)
      loss5 = loss5.mean()

      loss6, p_inds, n_inds, dist_ap, dist_an, dist_mat = global_loss(
        tri_loss, ff2, labels_t,
        normalize_feature=cfg.normalize_feature)
      '''

      '''
      ## cross loss ##
      cros1 = c_class1 * labab
      loss3 = -cros1.sum(dim=1)
      loss3 = loss3.mean()

      # loss1 = lossss(c_class, lab)

      #loss4, p_inds, n_inds, dist_ap, dist_an, dist_mat = global_loss(
      #  tri_loss, feat1, labels_t,
      #  normalize_feature=cfg.normalize_feature)


      ## cross loss ##
      cros2 = c_class2 * labab
      loss5 = -cros2.sum(dim=1)
      loss5 = loss5.mean()

      loss6, p_inds, n_inds, dist_ap, dist_an, dist_mat = global_loss(
        tri_loss, feat2, labels_t,
        normalize_feature=cfg.normalize_feature)
      '''
      '''
      ## cross loss ##
      cros3 = c_class3 * labab
      loss7 = -cros3.sum(dim=1)
      loss7 = loss7.mean()

      loss8, p_inds, n_inds, dist_ap, dist_an, dist_mat = global_loss(
        tri_loss, feat3, labels_t,
        normalize_feature=cfg.normalize_feature)
      '''
      loss = 0.5*loss9 + 0.5*loss10 + loss7 + loss8 + loss13 + loss14 + loss11 + loss12 + loss3 + loss4
      #loss = loss2

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      ############
      # Step Log #
      ############

      # precision
      prec = (dist_an > dist_ap).data.float().mean()
      # the proportion of triplets that satisfy margin
      sm = (dist_an > dist_ap + cfg.margin).data.float().mean()
      # average (anchor, positive) distance
      d_ap = dist_ap.data.mean()
      # average (anchor, negative) distance
      d_an = dist_an.data.mean()

      prec_meter.update(prec)
      sm_meter.update(sm)
      dist_ap_meter.update(d_ap)
      dist_an_meter.update(d_an)
      loss_meter.update(to_scalar(loss))

      if step % cfg.steps_per_log == 0:
        time_log = '\tStep {}/Ep {}, {:.2f}s'.format(
          step, ep + 1, time.time() - step_st, )

        tri_log = (', prec {:.2%}, sm {:.2%}, '
                   'd_ap {:.4f}, d_an {:.4f}, '
                   'loss {:.4f}'.format(
          prec_meter.val, sm_meter.val,
          dist_ap_meter.val, dist_an_meter.val,
          loss_meter.val, ))

        log = time_log + tri_log
        print(log)

    #############
    # Epoch Log #
    #############

    time_log = 'Ep {}, {:.2f}s'.format(ep + 1, time.time() - ep_st)

    tri_log = (', prec {:.2%}, sm {:.2%}, '
               'd_ap {:.4f}, d_an {:.4f}, '
               'loss {:.4f}'.format(
      prec_meter.avg, sm_meter.avg,
      dist_ap_meter.avg, dist_an_meter.avg,
      loss_meter.avg, ))

    log = time_log + tri_log
    print(log)

    ##########################
    # Test on Validation Set #
    ##########################

    mAP, Rank1 = 0, 0
    #if ((ep + 1) % cfg.epochs_per_val == 0) and (val_set is not None):
     # mAP, Rank1 = validate()
    if (ep + 1) % 50 == 0:
      ckptt = cfg.ckpt_file + str(ep)
      save_ckpt(modules_optims, ep + 1, 0, ckptt)
      test(load_model_weight=False)
    else:
      if ((ep + 1)>250) and ((ep + 1)<600) and (ep + 1) % 10 == 0:
        test(load_model_weight=False)


    # Log to TensorBoard

    if cfg.log_to_file:
      if writer is None:
        writer = SummaryWriter(log_dir=osp.join(cfg.exp_dir, 'tensorboard'))
      writer.add_scalars(
        'val scores',
        dict(mAP=mAP,
             Rank1=Rank1),
        ep)
      writer.add_scalars(
        'loss',
        dict(loss=loss_meter.avg, ),
        ep)
      writer.add_scalars(
        'precision',
        dict(precision=prec_meter.avg, ),
        ep)
      writer.add_scalars(
        'satisfy_margin',
        dict(satisfy_margin=sm_meter.avg, ),
        ep)
      writer.add_scalars(
        'average_distance',
        dict(dist_ap=dist_ap_meter.avg,
             dist_an=dist_an_meter.avg, ),
        ep)

    # save ckpt
    if cfg.log_to_file:
      save_ckpt(modules_optims, ep + 1, 0, cfg.ckpt_file)

  ########
  # Test #
  ########

  #test(load_model_weight=False)


if __name__ == '__main__':
  main()
