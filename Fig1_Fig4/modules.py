import torch.nn as nn
import torch
import matplotlib.pyplot as plt

def set_fr_module(args):
    """
    Create a frequency-representation module
    """
    net = None
    if args.fr_module_type == 'psnet':
        net = PSnet(signal_dim=args.signal_dim, fr_size=args.fr_size, n_filters=args.fr_n_filters,
                    inner_dim=args.fr_inner_dim, n_layers=args.fr_n_layers, kernel_size=args.fr_kernel_size)
    elif args.fr_module_type == 'fr':
        assert args.fr_size == args.fr_inner_dim * args.fr_upsampling, \
            'The desired size of the frequency representation (fr_size) must be equal to inner_dim*upsampling'
        net = FrequencyRepresentationModule(signal_dim=args.signal_dim, n_filters=args.fr_n_filters,
                                            inner_dim=args.fr_inner_dim, n_layers=args.fr_n_layers,
                                            upsampling=args.fr_upsampling, kernel_size=args.fr_kernel_size,
                                            kernel_out=args.fr_kernel_out)

    else:
        raise NotImplementedError('Frequency representation module type not implemented')
    if args.use_cuda:
        net.cuda()
    return net


def set_deepfreqRes_module(args):
    """
    Create a frequency-representation module
    """
    net = None
    if args.fr_module_type == 'psnet':
        net = PSnet(signal_dim=args.signal_dim, fr_size=args.fr_size, n_filters=args.fr_n_filters,
                    inner_dim=args.fr_inner_dim, n_layers=args.fr_n_layers, kernel_size=args.fr_kernel_size)
    elif args.fr_module_type == 'fr':
        assert args.fr_size == args.fr_inner_dim * args.fr_upsampling, \
            'The desired size of the frequency representation (fr_size) must be equal to inner_dim*upsampling'
        net = FrequencyRepresentationModule_deepfreqRes(signal_dim=args.signal_dim, n_filters=args.fr_n_filters,
                                            inner_dim=args.fr_inner_dim, n_layers=args.fr_n_layers,
                                            upsampling=args.fr_upsampling, kernel_size=args.fr_kernel_size,
                                            kernel_out=args.fr_kernel_out)

    else:
        raise NotImplementedError('Frequency representation module type not implemented')
    if args.use_cuda:
        net.cuda()
    return net

def set_freq_module(args):
    """
    Create a frequency-representation module
    """
    net = None
    if args.fr_module_type == 'psnet':
        net = PSnet(signal_dim=args.signal_dim, fr_size=args.fr_size, n_filters=args.fr_n_filters,
                    inner_dim=args.fr_inner_dim, n_layers=args.fr_n_layers, kernel_size=args.fr_kernel_size)
    elif args.fr_module_type == 'fr':
        assert args.fr_size == args.fr_inner_dim * args.fr_upsampling, \
            'The desired size of the frequency representation (fr_size) must be equal to inner_dim*upsampling'
        net = FrequencyRepresentationModule_freq(signal_dim=args.signal_dim, n_filters=args.fr_n_filters,
                                            inner_dim=args.fr_inner_dim, n_layers=args.fr_n_layers,
                                            upsampling=args.fr_upsampling, kernel_size=args.fr_kernel_size,
                                            kernel_out=args.fr_kernel_out)

    else:
        raise NotImplementedError('Frequency representation module type not implemented')
    if args.use_cuda:
        net.cuda()
    return net


def set_rdn_bias_module(args):
    """
    Create a frequency-representation module
    """
    net = None
    if args.fr_module_type == 'psnet':
        net = PSnet(signal_dim=args.signal_dim, fr_size=args.fr_size, n_filters=args.fr_n_filters,
                    inner_dim=args.fr_inner_dim, n_layers=args.fr_n_layers, kernel_size=args.fr_kernel_size)
    elif args.fr_module_type == 'fr':
        assert args.fr_size == args.fr_inner_dim * args.fr_upsampling, \
            'The desired size of the frequency representation (fr_size) must be equal to inner_dim*upsampling'
        net = FrequencyRepresentationModule_dense_bias(scale_factor=args.fr_upsampling, num_features=args.fr_n_filters,
                                            num_blocks=16, num_layers=8,
                                            growth_rate=args.fr_n_filters,signal_dim=args.signal_dim,inner_dim=args.fr_inner_dim)

    else:
        raise NotImplementedError('Frequency representation module type not implemented')
    if args.use_cuda:
        net.cuda()
    return net
def set_rdn_module(args):
    """
    Create a frequency-representation module
    """
    net = None
    if args.fr_module_type == 'psnet':
        net = PSnet(signal_dim=args.signal_dim, fr_size=args.fr_size, n_filters=args.fr_n_filters,
                    inner_dim=args.fr_inner_dim, n_layers=args.fr_n_layers, kernel_size=args.fr_kernel_size)
    elif args.fr_module_type == 'fr':
        assert args.fr_size == args.fr_inner_dim * args.fr_upsampling, \
            'The desired size of the frequency representation (fr_size) must be equal to inner_dim*upsampling'
        net = FrequencyRepresentationModule_dense(scale_factor=args.fr_upsampling, num_features=args.fr_n_filters,
                                            num_blocks=16, num_layers=8,
                                            growth_rate=args.fr_n_filters,signal_dim=args.signal_dim,inner_dim=args.fr_inner_dim)

    else:
        raise NotImplementedError('Frequency representation module type not implemented')
    if args.use_cuda:
        net.cuda()
    return net

def set_rdnamp_module(args):
    """
    Create a frequency-representation module
    """
    net = None
    if args.fr_module_type == 'psnet':
        net = PSnet(signal_dim=args.signal_dim, fr_size=args.fr_size, n_filters=args.fr_n_filters,
                    inner_dim=args.fr_inner_dim, n_layers=args.fr_n_layers, kernel_size=args.fr_kernel_size)
    elif args.fr_module_type == 'fr':
        assert args.fr_size == args.fr_inner_dim * args.fr_upsampling, \
            'The desired size of the frequency representation (fr_size) must be equal to inner_dim*upsampling'
        net = FrequencyRepresentationModule_dense_amp(scale_factor=args.fr_upsampling, num_features=args.fr_n_filters,
                                            num_blocks=16, num_layers=8,
                                            growth_rate=args.fr_n_filters,signal_dim=args.signal_dim,inner_dim=args.fr_inner_dim)

    else:
        raise NotImplementedError('Frequency representation module type not implemented')
    if args.use_cuda:
        net.cuda()
    return net

def set_rdnAt_module(args):
    """
    Create a frequency-representation module
    """
    net = None
    if args.fr_module_type == 'psnet':
        net = PSnet(signal_dim=args.signal_dim, fr_size=args.fr_size, n_filters=args.fr_n_filters,
                    inner_dim=args.fr_inner_dim, n_layers=args.fr_n_layers, kernel_size=args.fr_kernel_size)
    elif args.fr_module_type == 'fr':
        assert args.fr_size == args.fr_inner_dim * args.fr_upsampling, \
            'The desired size of the frequency representation (fr_size) must be equal to inner_dim*upsampling'
        net = FrequencyRepresentationModule_denseAttem(scale_factor=args.fr_upsampling, num_features=args.fr_n_filters,
                                            num_blocks=16, num_layers=8,
                                            growth_rate=args.fr_n_filters,signal_dim=args.signal_dim,inner_dim=args.fr_inner_dim)

    else:
        raise NotImplementedError('Frequency representation module type not implemented')
    if args.use_cuda:
        net.cuda()
    return net

def set_fc_module(args):
    """
    Create a frequency-counting module
    """
    assert args.fr_size % args.fc_downsampling == 0, \
        'The downsampling factor (fc_downsampling) does not divide the frequency representation size (fr_size)'
    net = None
    if args.fc_module_type == 'regression':
        net = FrequencyCountingModule(n_output=1, n_layers=args.fc_n_layers, n_filters=args.fc_n_filters,
                                      kernel_size=args.fc_kernel_size, fr_size=args.fr_size,
                                      downsampling=args.fc_downsampling, kernel_in=args.fc_kernel_in)
    elif args.fc_module_type == 'classification':
        net = FrequencyCountingModule(n_output=args.max_num_freq, n_layers=args.fc_n_layers,
                                      n_filters=args.fc_n_filters)
    else:
        NotImplementedError('Counter module type not implemented')
    if args.use_cuda:
        net.cuda()
    return net


class PSnet(nn.Module):
    def __init__(self, signal_dim=50, fr_size=1000, n_filters=8, inner_dim=100, n_layers=3, kernel_size=3):
        super().__init__()
        self.fr_size = fr_size
        self.num_filters = n_filters
        self.in_layer = nn.Linear(2 * signal_dim, inner_dim, bias=False)
        mod = []
        for n in range(n_layers):
            in_filters = n_filters if n > 0 else 1
            mod += [
                nn.Conv1d(in_channels=in_filters, out_channels=n_filters, kernel_size=kernel_size,
                          stride=1, padding=kernel_size // 2, bias=False),
                nn.BatchNorm1d(n_filters),
                nn.ReLU()
            ]
        self.mod = nn.Sequential(*mod)
        self.out_layer = nn.Linear(inner_dim * n_filters, fr_size, bias=True)

    def forward(self, inp):
        bsz = inp.size(0)
        inp = inp.view(bsz, -1)
        x = self.in_layer(inp).view(bsz, 1, -1)
        x = self.mod(x).view(bsz, -1)
        output = self.out_layer(x)
        return output


import torch


class FrequencyRepresentationModule_freq(nn.Module):

    def __init__(self, signal_dim=50, n_filters=8, n_layers=3, inner_dim=125,
                 kernel_size=3, upsampling=8, kernel_out=25):
        super().__init__()
        self.fr_size = inner_dim * upsampling
        self.n_filters = n_filters
        self.in_layer = nn.Linear(inner_dim, inner_dim * n_filters, bias=False)

        mod=[]
        for i in range(24):
            tmp = []
            tmp += [
                nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size //2, bias=False,
                          padding_mode='circular'),
                nn.BatchNorm1d(n_filters),
                nn.ReLU(),
                nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size //2, bias=False,
                          padding_mode='circular'),
                nn.BatchNorm1d(n_filters),
            ]
            mod+= [nn.Sequential(*tmp)]
        self.mod=nn.Sequential(*mod)
        activate_layer = []
        for i in range(24):
            activate_layer+=[nn.ReLU()]
        self.activate_layer=nn.Sequential(*activate_layer)
        self.out_layer = nn.ConvTranspose1d(n_filters, 1, 18, stride=16,
                                            padding=1, output_padding=0, bias=False)




    def forward(self, inp):
        bsz = inp.size(0)

        x = self.in_layer(inp).view(bsz, self.n_filters, -1)

        for i in range(24):
            res_x = self.mod[i](x)
            x = res_x + x
            x = self.activate_layer[i](x)
        x = self.out_layer(x).view(bsz, -1)
        return x


class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1= nn.Conv1d(in_planes, in_planes // 4, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2= nn.Conv1d(in_planes // 4, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])

        # local feature fusion
        self.lff = nn.Conv1d(in_channels + growth_rate * num_layers, growth_rate, kernel_size=1)
        # self.ca=ChannelAttention(growth_rate)
        # self.sa=SpatialAttention()
    # RDN
    def forward(self, x):
        return x + self.lff(self.layers(x))  # local residual learning

    # RDN+ATTENTION
    # def forward(self, x):
    #     F=self.lff(self.layers(x))
    #     Fbar=self.ca(F)*F
    #     return x + self.sa(Fbar)*Fbar
class FrequencyRepresentationModule_denseAttem(nn.Module):
    def __init__(self, scale_factor, num_features, growth_rate, num_blocks, num_layers,signal_dim,inner_dim):
        super().__init__()
        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers

        # shallow feature extraction
        self.fr_size = inner_dim * scale_factor
        self.n_filters = num_features
        self.in_layer = nn.Linear(2 * signal_dim, inner_dim * num_features, bias=False)
        # self.in_layer2 = nn.Conv1d(num_features, num_features, kernel_size=3,padding=3//2,bias=False)
        self.ca=ChannelAttention(growth_rate)
        self.sa=SpatialAttention()

        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB(self.G, self.G, self.C))

        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv1d(self.G * self.D, self.G0, kernel_size=1),
            nn.Conv1d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )

        # up-sampling
        # scale_factor=int(scale_factor/2)
        # assert 2 <= scale_factor <= 4
        # if scale_factor == 2 or scale_factor == 4:
        #     self.upscale = []
        #     for _ in range(scale_factor // 2):
        #         self.upscale.extend([nn.Conv1d(self.G0, self.G0 * (2 ** 2), kernel_size=3, padding=3 // 2),
        #                              nn.PixelShuffle(2)])
        #     self.upscale = nn.Sequential(*self.upscale)
        # else:
        #     self.upscale = nn.Sequential(
        #         nn.Conv1d(self.G0, self.G0 * (scale_factor ** 2), kernel_size=3, padding=3 // 2),
        #         nn.PixelShuffle(scale_factor)
        #     )

        # self.output = nn.Conv1d(self.G0, 1, kernel_size=3, padding=3 // 2)
        self.output = nn.ConvTranspose1d(self.G0, 1, 9, stride=8,
                                            padding=(9 - 8 + 1) // 2, output_padding=1, bias=False)

    def forward(self, x):
        bsz = x.size(0)
        inp = x.view(bsz, -1)
        sfe1 = self.in_layer(inp).view(bsz, self.n_filters, -1)
        # x = sfe1

        Fbar=self.ca(sfe1)*sfe1
        x= sfe1 + self.sa(Fbar)*Fbar

        local_features = []
        for i in range(self.D):
            x = self.rdbs[i](x)
            local_features.append(x)

        x = self.gff(torch.cat(local_features, 1)) + sfe1  # global residual learning
        # x = self.upscale(x)
        x = self.output(x).view(bsz, -1)
        return x

class FrequencyRepresentationModule_dense_amp(nn.Module):
    def __init__(self, scale_factor, num_features, growth_rate, num_blocks, num_layers,signal_dim,inner_dim):
        super().__init__()
        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers

        # shallow feature extraction
        self.fr_size = inner_dim * scale_factor
        self.n_filters = num_features
        self.in_layer = nn.Linear(2 * signal_dim, inner_dim * num_features, bias=False)
        # self.in_layer2 = nn.Conv1d(num_features, num_features, kernel_size=3,padding=3//2,bias=False)
        # self.ca=ChannelAttention(growth_rate)
        # self.sa=SpatialAttention()

        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB(self.G, self.G, self.C))

        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv1d(self.G * self.D, self.G0, kernel_size=1),
            nn.Conv1d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )

        self.output = nn.ConvTranspose1d(self.G0, 1, 9, stride=8,
                                            padding=(9 - 8 + 1) // 2, output_padding=1, bias=False)

    def forward(self, x,amp):
        bsz = x.size(0)
        inp = x.view(bsz, -1)
        sfe1 = self.in_layer(inp).view(bsz, self.n_filters, -1)
        x = sfe1

        # Fbar=self.ca(sfe1)*sfe1
        # x= sfe1 + self.sa(Fbar)*Fbar

        local_features = []
        for i in range(self.D):
            x = self.rdbs[i](x)
            local_features.append(x)

        x = self.gff(torch.cat(local_features, 1)) + sfe1  # global residual learning
        # x = self.upscale(x)
        x = self.output(x).view(bsz, -1)*amp
        return x

class FrequencyRepresentationModule_dense(nn.Module):
    def __init__(self, scale_factor, num_features, growth_rate, num_blocks, num_layers,signal_dim,inner_dim):
        super().__init__()
        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers

        # shallow feature extraction
        self.fr_size = inner_dim * scale_factor
        self.n_filters = num_features
        # self.in_layer = nn.Linear(2 * signal_dim, inner_dim * num_features, bias=False)
        self.in_layer = nn.Conv1d(1, num_features * inner_dim, kernel_size=(1, 128), padding=0, bias=False)
        self.in_layer2 = nn.Conv1d(num_features, num_features, kernel_size=3,padding=3//2,bias=False)
        # self.ca=ChannelAttention(growth_rate)
        # self.sa=SpatialAttention()

        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB(self.G, self.G, self.C))

        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv1d(self.G * self.D, self.G0, kernel_size=1),
            nn.Conv1d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )
        self.gff2 = nn.Sequential(
            nn.Conv1d(self.G * 2, self.G0, kernel_size=1),
            nn.Conv1d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )


        self.output = nn.ConvTranspose1d(self.G0, 1, 9, stride=8,
                                            padding=1, output_padding=1, bias=False)

    def forward(self, x):
        bsz = x.size(0)
        # inp = x.view(bsz, -1)
        inp = x.view(bsz, 1,-1)
        inp = inp.view(bsz, 1,1, 128)
        sfe1 = self.in_layer(inp).view(bsz, self.n_filters, -1)
        sfe2=self.in_layer2(sfe1)
        x = sfe2
        # x=self.mid(x)
        # for i in range(32):
        #     plt.ion()
        #     plt.plot(x[0,i])
        #     plt.show()
        #     plt.pause(1)

        # Fbar=self.ca(sfe1)*sfe1
        # x= sfe1 + self.sa(Fbar)*Fbar

        local_features = []
        for i in range(self.D):
            x = self.rdbs[i](x)
            # plt.figure()
            # for i in range(32):
            #     plt.ion()
            #     plt.plot(x[0, i])
            #     plt.show()
            #     plt.pause(1)
            local_features.append(x)

        # plt.figure()
        # for i in range(32):
        #     plt.ion()
        #     plt.plot(local_features[-1][0, i])
        #     plt.show()
        #     plt.pause(1)
        # fx = self.gff(torch.cat(local_features, 1))   # global residual learning
        # for i in range(32):
        #     plt.ion()
        #     plt.plot(fx[0,i])
        #     plt.show()
        #     plt.pause(1)
        # fx+= sfe1

        # comd=[]
        # comd.append(local_features[-1])
        # comd.append(sfe1)
        # fx = self.gff2(torch.cat(comd, 1))
        # plt.figure()
        # for i in range(32):
        #     plt.ion()
        #     plt.plot(x[0, i])
        #     plt.show()
        #     plt.pause(1)
        x=self.output(local_features[-1]).view(bsz,-1)
        # x1=x[:,:,0:2]
        # x2=x[:,:,-1]
        # plt.figure()
        # plt.plot(x[0,:])
        return x
class FrequencyRepresentationModule_dense_bias(nn.Module):
    def __init__(self, scale_factor, num_features, growth_rate, num_blocks, num_layers,signal_dim,inner_dim):
        super().__init__()
        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers

        # shallow feature extraction
        self.fr_size = inner_dim * scale_factor
        self.n_filters = num_features
        self.in_layer = nn.Linear(2* signal_dim, inner_dim * num_features, bias=False)
        # self.in_layer2 = nn.Conv1d(num_features, num_features, kernel_size=3,padding=3//2,bias=False)
        # self.ca=ChannelAttention(growth_rate)
        # self.sa=SpatialAttention()

        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB(self.G, self.G, self.C))

        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv1d(self.G * self.D, self.G0, kernel_size=1),
            nn.Conv1d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )

        self.output = nn.ConvTranspose1d(self.G0, 1, 9, stride=8,
                                            padding=(9 - 8 + 1) // 2, output_padding=1, bias=True)



    def forward(self, x):
        bsz = x.size(0)
        inp = x.view(bsz, -1)
        sfe1 = self.in_layer(inp).view(bsz, self.n_filters, -1)
        x = sfe1

        # Fbar=self.ca(sfe1)*sfe1
        # x= sfe1 + self.sa(Fbar)*Fbar

        local_features = []
        for i in range(self.D):
            x = self.rdbs[i](x)
            local_features.append(x)

        x = self.gff(torch.cat(local_features, 1)) + sfe1  # global residual learning
        # x = self.upscale(x)
        x = self.output(x).view(bsz, -1)
        return x


class FrequencyRepresentationModule_deepfreqRes(nn.Module):
    # pre_path = 'checkpoint_2D_2/experiment/pre/epoch_160-skip-conn.pth'
    def __init__(self, signal_dim=50, n_filters=8, n_layers=3, inner_dim=125,
                 kernel_size=3, upsampling=8, kernel_out=25):
        super().__init__()
        self.fr_size = inner_dim * upsampling
        self.n_filters = n_filters
        self.in_layer = nn.Linear(2 * signal_dim, inner_dim * n_filters, bias=False)
        # self.in_layer = nn.Conv1d(2, n_filters * inner_dim, kernel_size=(1, 64), padding=0, bias=False)
        # self.mid1 = nn.ConvTranspose1d(n_filters, n_filters, kernel_size, stride=2,
        #                                     padding=(kernel_size - 2 + 1) // 2, output_padding=1, bias=False)
        self.n_layer=n_layers
        mod=[]
        for i in range(n_layers):
            tmp = []
            tmp += [
                nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size //2, bias=False,
                          padding_mode='circular'),
                nn.BatchNorm1d(n_filters),
                nn.ReLU(),
                nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size //2, bias=False,
                          padding_mode='circular'),
                nn.BatchNorm1d(n_filters),
            ]
            mod+= [nn.Sequential(*tmp)]
        self.mod=nn.Sequential(*mod)
        activate_layer = []
        for i in range(n_layers):
            activate_layer+=[nn.ReLU()]
        self.activate_layer=nn.Sequential(*activate_layer)
        self.out_layer = nn.ConvTranspose1d(n_filters, 1, 18, stride=16,
                                            padding=1, output_padding=0, bias=False)




    def forward(self, inp):
        bsz = inp.size(0)
        inp = inp.view(bsz, -1)
        # inp=inp[:,:,None,:]
        x = self.in_layer(inp).view(bsz, self.n_filters, -1)
        # x=self.mid1(x)
        for i in range(self.n_layer):
            res_x = self.mod[i](x)
            x = res_x + x
            x = self.activate_layer[i](x)
        x = self.out_layer(x).view(bsz, -1)
        return x

class FrequencyRepresentationModule(nn.Module):
    def __init__(self, signal_dim=50, n_filters=8, n_layers=3, inner_dim=125,
                 kernel_size=3, upsampling=8, kernel_out=25):
        super().__init__()
        self.fr_size = inner_dim * upsampling
        self.n_filters = n_filters
        self.in_layer = nn.Linear(2 * signal_dim, inner_dim * n_filters, bias=False)
        mod = []
        for n in range(n_layers):
            mod += [
                nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size //2, bias=False,
                          padding_mode='circular'),
                nn.BatchNorm1d(n_filters),
                nn.ReLU(),
            ]
            self.mod = nn.Sequential(*mod)
        self.out_layer = nn.ConvTranspose1d(n_filters, 1, kernel_out, stride=upsampling,
                                            padding=(kernel_out - upsampling + 1) // 2, output_padding=1, bias=False)

    def forward(self, inp):
        bsz = inp.size(0)
        inp = inp.view(bsz, -1)
        x = self.in_layer(inp).view(bsz, self.n_filters, -1)
        x = self.mod(x)
        # plt.plot(x[0,0,:].cpu().detach().numpy())
        # plt.close()
        x = self.out_layer(x).view(bsz, -1)
        return x


class FrequencyCountingModule(nn.Module):
    def __init__(self, n_output, n_layers, n_filters, kernel_size, fr_size, downsampling, kernel_in):
        super().__init__()
        mod = [nn.Conv1d(1, n_filters, kernel_in, stride=downsampling, padding=kernel_in - downsampling,
                             padding_mode='circular')]
        for i in range(n_layers):
            mod += [
                nn.Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding=kernel_size - 1, bias=False,
                          padding_mode='circular'),
                nn.BatchNorm1d(n_filters),
                nn.ReLU(),
            ]
        mod += [nn.Conv1d(n_filters, 1, 1)]
        self.mod = nn.Sequential(*mod)
        self.out_layer = nn.Linear(fr_size // downsampling, n_output)

    def forward(self, inp):
        bsz = inp.size(0)
        inp = inp[:, None]
        x = self.mod(inp)
        x = x.view(bsz, -1)
        y = self.out_layer(x)
        return y
