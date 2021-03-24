import os
import sys
import time
import argparse
import logging
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from data import dataset
import complexModules
import util
from data.noise import noise_torch
from data import fr
from data.loss import fnr
from torch import nn
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)


def orthogonal_regularization(model, device, beta=1e-4):
    # beta * (||W^T.W * (1-I)||_F)^2 or
    # beta * (||W.W.T * (1-I)||_F)^2
    # 若 H < W,可以使用前者， 若 H > W, 可以使用后者，这样可以适当减少内存
    loss_orth = torch.tensor(0., dtype=torch.float32, device=device)
    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad and len(param.shape) == 4:
            # 是weight，而不是bias
            # 当然是指定被训练的参数
            # 只对卷积层参数做这样的正则化，而不包括嵌入层（维度是2）等。
            N, C, H, W = param.shape
            weight = param.view(N * C, H, W)
            weight_squared = torch.bmm(weight, weight.permute(0, 2, 1))  # (N * C) * H * H
            ones = torch.ones(N * C, H, H, dtype=torch.float32)  # (N * C) * H * H
            diag = torch.eye(H, dtype=torch.float32)  # (N * C) * H * H
            loss_orth += ((weight_squared * (ones - diag).to(device)) ** 2).sum()

    return loss_orth * beta

class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes = 3, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """

        super(focal_loss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            print("Focal_loss alpha = {}, 将对每一类权重进行精细化赋值".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            # print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1,preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds, dim=1) # 这里并没有直接使用log_softmax, 因为后面会用到softmax的结果(当然你也可以使用log_softmax,然后进行exp操作)
        preds_logsoft = torch.log(preds_softmax)
        preds_softmax = preds_softmax.gather(1,labels.view(-1,1).long())   # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1).long())
        self.alpha = self.alpha.gather(0,labels.view(-1).long())
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


def train_frequency_representation(args, fr_module, fr_optimizer, fr_criterion, fr_scheduler, train_loader, val_loader,
                                   xgrid, epoch, tb_writer):
    """
    Train the frequency-representation module for one epoch
    """
    epoch_start_time = time.time()
    fr_module.train()
    # adaptive.train()
    loss_train_fr,fnr_train = 0,0

    lr_mult = 1.2
    lr = []
    losses = []
    best_loss = 1e12
    for batch_idx, (clean_signal, target_fr, freq,fr_ground,m1,m2) in enumerate(train_loader):
        if args.use_cuda:
            clean_signal, target_fr,fr_ground,m1,m2 = clean_signal.cuda(), target_fr.cuda(),fr_ground.cuda(),m1.cuda(),m2.cuda()
        noisy_signal = noise_torch(clean_signal, args.snr, args.noise)
        # sig_win = torch.zeros(noisy_signal.size()[0], 2, noisy_signal.size()[2]).cuda()
        # win = torch.from_numpy(np.hamming(noisy_signal.size()[2]).astype('float32')).cuda()
        for i in range(noisy_signal.size()[0]):
            mv=torch.max(torch.sqrt((pow(noisy_signal[i][0],2)+pow(noisy_signal[i][1],2))))
            noisy_signal[i][0]=noisy_signal[i][0]/mv
            noisy_signal[i][1]=noisy_signal[i][1]/mv
            # sig_win[i,0]=noisy_signal[i][0]*win
            # sig_win[i, 1] = noisy_signal[i][1] * win
            # mv2=torch.max(torch.sqrt(torch.pow(sig_win[i][0],2)+torch.pow(sig_win[i][1],2)))
            # sig_win[i][0]=sig_win[i][0]/mv2
            # sig_win[i][1]=sig_win[i][1]/mv2

        output_fr = fr_module(noisy_signal)
        # if epoch==90:
        #     plt.figure()
        #     plt.plot(target_fr[0].cpu().detach().numpy())
        #     plt.plot(output_fr[0].cpu().detach().numpy())
        #     x=1
        loss_fr = torch.pow(((output_fr) - (target_fr)),2)

        loss_fr = torch.sum(loss_fr).to(torch.float32)
        # nfreq = (freq >= -0.5).sum(dim=1)
        # f_hat = fr.find_freq(output_fr.cpu().detach().numpy(), nfreq, xgrid)
        # fnr_train += fnr(f_hat, freq.cpu().numpy(), args.signal_dim)
        # loss_fr+=fnr_train
        # l1_regularization, l2_regularization = torch.tensor([0], dtype=torch.float32).cuda(), torch.tensor([0],
        #                                                                                             dtype=torch.float32).cuda()
        # wgt_mat = torch.zeros(4096,64).cuda()
        # for name,param in fr_module.named_parameters():
        #     if 'in_layer.' in name:
        #         wgt_mat+=torch.pow(param,2)
        # wgt_mat = torch.sqrt(wgt_mat)
        # l1_regularization += torch.norm(wgt_mat, 1)  # L1正则化
        # l2_regularization += torch.norm(wgt_mat, 2)  # L2 正则化
        #
        #
        # H, W = wgt_mat.shape
        # weight=wgt_mat
        # weight_squared = torch.matmul(weight, weight.permute(1, 0))  # H * H
        # ones = torch.ones(H, H, dtype=torch.float32)  # (N * C) * H * H
        # diag = torch.eye(H, dtype=torch.float32)  # (N * C) * H * H
        # loss_fr += ((weight_squared * (ones - diag).cuda()) ** 2).sum()

        fr_optimizer.zero_grad()
        loss_fr.backward()
        fr_optimizer.step()
        loss_train_fr += loss_fr.data.item()


    # mxnet
    #     lr.append(fr_optimizer.param_groups[0]['lr'])
    #     losses.append(loss_fr.data.item())
    #     fr_optimizer.param_groups[0]['lr']=(fr_optimizer.param_groups[0]['lr'] * lr_mult)
    #
    #     if loss_fr.data.item() < best_loss:
    #         best_loss = loss_fr.data.item()
    #
    #     if loss_fr.data.item() > 4 * best_loss or fr_optimizer.param_groups[0]['lr'] > 1.:
    #         break
    #
    # plt.plot(lr, losses)
    # plt.show()


    fr_module.eval()
    loss_val_fr, fnr_val = 0, 0
    for batch_idx, (noisy_signal, _, target_fr, freq,fr_ground,m1,m2) in enumerate(val_loader):
        if args.use_cuda:
            noisy_signal, target_fr,fr_ground,m1,m2 = noisy_signal.cuda(), target_fr.cuda(),fr_ground.cuda(),m1.cuda(),m2.cuda()
        # sig_win = torch.zeros(noisy_signal.size()[0], 2, noisy_signal.size()[2]).cuda()
        # win = torch.from_numpy(np.hamming(noisy_signal.size()[2]).astype('float32')).cuda()
        for i in range(noisy_signal.size()[0]):
            mv=torch.max(torch.sqrt((pow(noisy_signal[i][0],2)+pow(noisy_signal[i][1],2))))
            noisy_signal[i][0]=noisy_signal[i][0]/mv
            noisy_signal[i][1]=noisy_signal[i][1]/mv
            # sig_win[i,0]=noisy_signal[i][0]*win
            # sig_win[i, 1] = noisy_signal[i][1] * win
            # mv2=torch.max(torch.sqrt(torch.pow(sig_win[i][0],2)+torch.pow(sig_win[i][1],2)))
            # sig_win[i][0]=sig_win[i][0]/mv2
            # sig_win[i][1]=sig_win[i][1]/mv2

        with torch.no_grad():
            output_fr = fr_module(noisy_signal)
        # plt.figure()
        # plt.plot(target_fr[0].cpu().detach().numpy())
        # plt.plot(output_fr[0].cpu().detach().numpy())

        loss_fr = torch.pow(((output_fr) - (target_fr)),2)

        loss_fr = torch.sum(loss_fr).to(torch.float32)

        # l1_regularization, l2_regularization = torch.tensor([0], dtype=torch.float32).cuda(), torch.tensor([0],
        #                                                                                             dtype=torch.float32).cuda()  # 定义L1及L2正则化损失
        #
        # wgt_mat = torch.zeros(4096, 64).cuda()
        # for name, param in fr_module.named_parameters():
        #     if 'in_layer.' in name:
        #         wgt_mat += torch.pow(param, 2)
        # wgt_mat = torch.sqrt(wgt_mat)
        # l1_regularization += torch.norm(wgt_mat, 1)  # L1正则化
        # l2_regularization += torch.norm(wgt_mat, 2)  # L2 正则化
        #
        # H, W = wgt_mat.shape
        # weight = wgt_mat
        # weight_squared = torch.matmul(weight, weight.permute(1, 0))  #  H * H
        # ones = torch.ones( H, H, dtype=torch.float32)  # (N * C) * H * H
        # diag = torch.eye(H, dtype=torch.float32)  # (N * C) * H * H
        # loss_fr += ((weight_squared * (ones - diag).cuda()) ** 2).sum()
        # # loss_fr+=l2_regularization[0]

        loss_val_fr += loss_fr.data.item()
        nfreq = (freq >= -0.5).sum(dim=1)
        f_hat = fr.find_freq(output_fr.cpu().detach().numpy(), nfreq, xgrid)
        fnr_val += fnr(f_hat, freq.cpu().numpy(), args.signal_dim)

    loss_train_fr /= args.n_training
    loss_val_fr /= args.n_validation
    fnr_val *= 100 / args.n_validation

    tb_writer.add_scalar('fr_l2_training', loss_train_fr, epoch)
    tb_writer.add_scalar('fr_l2_validation', loss_val_fr, epoch)
    tb_writer.add_scalar('fr_FNR', fnr_val, epoch)

    fr_scheduler.step(loss_val_fr)
    logger.info("Epochs: %d / %d, Time: %.1f, FR training L2 loss %.2f, FR validation L2 loss %.2f, FNR %.2f %%",
                epoch, args.n_epochs_fr, time.time() - epoch_start_time, loss_train_fr, loss_val_fr,
                fnr_val)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # basic parameters
    parser.add_argument('--output_dir', type=str, default='./checkpoint/skipfreq_snr_big8', help='output directory')
    parser.add_argument('--no_cuda', action='store_true', help="avoid using CUDA when available")
    # dataset parameters
    parser.add_argument('--batch_size', type=int, default=64, help='batch size used during training')
    parser.add_argument('--signal_dim', type=int, default=64, help='dimensionof the input signal')
    parser.add_argument('--fr_size', type=int, default=4096, help='size of the frequency representation')
    parser.add_argument('--max_n_freq', type=int, default=10,
                        help='for each signal the number of frequencies is uniformly drawn between 1 and max_n_freq')
    parser.add_argument('--min_sep', type=float, default=0.5,
                        help='minimum separation between spikes, normalized by signal_dim')
    parser.add_argument('--distance', type=str, default='normal', help='distance distribution between spikes')
    parser.add_argument('--amplitude', type=str, default='uniform', help='spike amplitude distribution')
    parser.add_argument('--floor_amplitude', type=float, default=0.1, help='minimum amplitude of spikes')
    parser.add_argument('--noise', type=str, default='gaussian_blind', help='kind of noise to use')
    parser.add_argument('--snr', type=float, default=-10, help='snr parameter')
    # frequency-representation (fr) module parameters
    parser.add_argument('--fr_module_type', type=str, default='fr', help='type of the fr module: [fr | psnet]')
    parser.add_argument('--fr_n_layers', type=int, default=24, help='number of convolutional layers in the fr module')
    parser.add_argument('--fr_n_filters', type=int, default=32, help='number of filters per layer in the fr module')
    parser.add_argument('--fr_kernel_size', type=int, default=3,
                        help='filter size in the convolutional blocks of the fr module')
    parser.add_argument('--fr_kernel_out', type=int, default=25, help='size of the conv transpose kernel')
    parser.add_argument('--fr_inner_dim', type=int, default=256, help='dimension after first linear transformation')
    parser.add_argument('--fr_upsampling', type=int, default=16,
                        help='stride of the transposed convolution, upsampling * inner_dim = fr_size')

    # kernel parameters used to generate the ideal frequency representation
    parser.add_argument('--kernel_type', type=str, default='gaussian',
                        help='type of kernel used to create the ideal frequency representation [gaussian, triangle or closest]')
    parser.add_argument('--triangle_slope', type=float, default=4000,
                        help='slope of the triangle kernel normalized by signal_dim')
    parser.add_argument('--gaussian_std', type=float, default=0.12,
                        help='std of the gaussian kernel normalized by signal_dim')
    # training parameters
    parser.add_argument('--n_training', type=int, default=50000, help='# of training data')
    parser.add_argument('--n_validation', type=int, default=1000, help='# of validation data')
    parser.add_argument('--lr_fr', type=float, default=0.003,
                        help='initial learning rate for adam optimizer used for the frequency-representation module')
    parser.add_argument('--n_epochs_fr', type=int, default=410, help='number of epochs used to train the fr module')
    parser.add_argument('--save_epoch_freq', type=int, default=10,
                        help='frequency of saving checkpoints at the end of epochs')
    parser.add_argument('--numpy_seed', type=int, default=100)
    parser.add_argument('--torch_seed', type=int, default=76)

    args = parser.parse_args()


    if torch.cuda.is_available() and not args.no_cuda:
        args.use_cuda = True
    else:
        args.use_cuda = False

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    file_handler = logging.FileHandler(filename=os.path.join(args.output_dir, 'run.log'))
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]
    logging.basicConfig(
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=handlers
    )

    tb_writer = SummaryWriter(args.output_dir)
    util.print_args(logger, args)

    np.random.seed(args.numpy_seed)
    torch.manual_seed(args.torch_seed)

    train_loader = dataset.make_train_data(args)
    val_loader = dataset.make_eval_data(args)


    fr_module = complexModules.set_skip_module(args)
    fr_optimizer, fr_scheduler = util.set_optim(args, fr_module, 'skip')
    fr_criterion = torch.nn.MSELoss(reduction='sum')
    start_epoch = 1

    logger.info('[Network] Number of parameters in the frequency-representation module : %.3f M' % (
                util.model_parameters(fr_module) / 1e6))


    xgrid = np.linspace(-0.5, 0.5, args.fr_size, endpoint=False)
    for epoch in range(start_epoch, args.n_epochs_fr + 1):

        if epoch < args.n_epochs_fr:
            train_frequency_representation(args=args, fr_module=fr_module, fr_optimizer=fr_optimizer, fr_criterion=fr_criterion,
                                           fr_scheduler=fr_scheduler, train_loader=train_loader, val_loader=val_loader,
                                           xgrid=xgrid, epoch=epoch, tb_writer=tb_writer)


        if epoch % args.save_epoch_freq == 0 or epoch == args.n_epochs_fr :
            util.save(fr_module, fr_optimizer, fr_scheduler, args, epoch, args.fr_module_type)

