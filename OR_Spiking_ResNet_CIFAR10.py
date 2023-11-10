import sys
import torch
import torch.nn as nn
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
sys.path.append("..")
from spikingjelly.activation_based import neuron, functional, surrogate, layer, monitor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.cuda import amp
import os
import random
# Multi GPU
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Fixed random seed
torch.manual_seed(2023)  # Set seeds for CPU and CUDA to generate random numbers and fix the results
np.random.seed(2023)
torch.backends.cudnn.deterministic = True  # Enable Cuda to use the same core allocation method
torch.backends.cudnn.benchmark = False  # No hardware level optimization for convolution and other operations
random.seed(2023)

class MA(nn.Module):
    def __init__(self, T: int, C: int, reduction_t: int = 8, reduction_c: int = 8, kernel_size=3):
        """
        :param T: Time step of input data
        :param C: Number of channels for input data
        :param reduction_t: Time compression ratio
        :param reduction_c: Channel compression ratio
        :param kernel_size: Convolutional Kernel Size of Spatial Attention Mechanism (3/5)
        :param v_reset_max:reset max
        :param v_reset_min:reset min
        :param decay:How much does reset decay every time there is data input
        The input size is' [T, N, C, H, W] ', and after passing through the MultiStepMultiDimensionalAttention layer, the output is' [T, N, C, H, W]'.
        """
        super().__init__()

        assert T >= reduction_t, 'reduction_t cannot be greater than T'
        assert C >= reduction_c, 'reduction_c cannot be greater than C'



        from einops import rearrange

        # Attention
        class TimeAttention(nn.Module):
            def __init__(self, in_planes, ratio=16):
                super(TimeAttention, self).__init__()
                self.avg_pool = nn.AdaptiveAvgPool3d(1)
                self.max_pool = nn.AdaptiveMaxPool3d(1)
                self.sharedMLP = nn.Sequential(
                    nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False),
                    nn.ReLU(),
                    nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False),
                ).cuda()
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                avgout = self.sharedMLP(self.avg_pool(x))
                maxout = self.sharedMLP(self.max_pool(x))
                return self.sigmoid(avgout + maxout)


        class ChannelAttention(nn.Module):
            def __init__(self, in_planes, ratio=16):
                super(ChannelAttention, self).__init__()
                self.avg_pool = nn.AdaptiveAvgPool3d(1)
                self.max_pool = nn.AdaptiveMaxPool3d(1)
                self.sharedMLP = nn.Sequential(
                    nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False),
                    nn.ReLU(),
                    nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False),
                ).cuda()
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x = rearrange(x, "b f c h w -> b c f h w")
                avgout = self.sharedMLP(self.avg_pool(x))
                maxout = self.sharedMLP(self.max_pool(x))
                out = self.sigmoid(avgout + maxout)
                out = rearrange(out, "b c f h w -> b f c h w")
                return out

        class SpatialAttention(nn.Module):
            def __init__(self, kernel_size=3):
                super(SpatialAttention, self).__init__()
                assert kernel_size in (3, 7), "kernel size must be 3 or 7"
                padding = 3 if kernel_size == 7 else 1
                self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False).cuda()
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x = rearrange(x, "b f c h w -> b (f c) h w")
                avgout = torch.mean(x, dim=1, keepdim=True)
                maxout, _ = torch.max(x, dim=1, keepdim=True)
                x = torch.cat([avgout, maxout], dim=1)
                x = self.conv(x)
                x = x.unsqueeze(1)
                return self.sigmoid(x)

        self.ta = TimeAttention(T, reduction_t)
        # self.ca = ChannelAttention(C, reduction_c)
        # self.sa = SpatialAttention(kernel_size)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        """
        In the case of mixed accuracy, when jumping from training to testing, the type of reset is float16, and the type of x is float32
        :param x: Input to the network
        :return:
        """
        assert x.dim() == 5, ValueError(
            f'expected 5D input with shape [T, N, C, H, W], but got input with shape {x.shape}')

        x = x.transpose(0, 1)
        ta = self.ta(x)
        # ca = self.ca(x)
        # sa = self.sa(x)
        out = ta * x
        # out = ca * x
        # out = sa * x
        out = self.relu(out)
        out = out.transpose(0, 1)


        return out


class IA(nn.Module):
    def __init__(self, T: int, C: int, reduction_t: int = 8, reduction_c: int = 8, kernel_size=3):
        """
        :param T: Time step of input data
        :param C: Number of channels for input data
        :param reduction_t: Time compression ratio
        :param reduction_c: Channel compression ratio
        :param kernel_size: Convolutional Kernel Size of Spatial Attention Mechanism (3/5)
        :param v_reset_max:reset max
        :param v_reset_min:reset min
        :param decay:How much does reset decay every time there is data input
        The input size is' [T, N, C, H, W] ', and after passing through the MultiStepMultiDimensionalAttention layer, the output is' [T, N, C, H, W]'.
        """
        super().__init__()

        assert T >= reduction_t, 'reduction_t cannot be greater than T'
        assert C >= reduction_c, 'reduction_c cannot be greater than C'



        from einops import rearrange

        # Attention
        class TimeAttention(nn.Module):
            def __init__(self, in_planes, ratio=16):
                super(TimeAttention, self).__init__()
                self.avg_pool = nn.AdaptiveAvgPool3d(1)
                self.max_pool = nn.AdaptiveMaxPool3d(1)
                self.sharedMLP = nn.Sequential(
                    nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False),
                    nn.ReLU(),
                    nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False),
                ).cuda()

                self.lif = nn.Sequential(
                    neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0,
                                   surrogate_function=surrogate.ATan()),
                )

            def forward(self, x):
                avgout = self.sharedMLP(self.avg_pool(x))
                maxout = self.sharedMLP(self.max_pool(x))
                return self.lif(avgout + maxout)


        class ChannelAttention(nn.Module):
            def __init__(self, in_planes, ratio=16):
                super(ChannelAttention, self).__init__()
                self.avg_pool = nn.AdaptiveAvgPool3d(1)
                self.max_pool = nn.AdaptiveMaxPool3d(1)
                self.sharedMLP = nn.Sequential(
                    nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False),
                    nn.ReLU(),
                    nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False),
                ).cuda()
                self.lif = nn.Sequential(
                    neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0,
                                   surrogate_function=surrogate.ATan()),
                )

            def forward(self, x):
                x = rearrange(x, "b f c h w -> b c f h w")
                avgout = self.sharedMLP(self.avg_pool(x))
                maxout = self.sharedMLP(self.max_pool(x))
                out = self.lif(avgout + maxout)
                out = rearrange(out, "b c f h w -> b f c h w")
                return out

        class SpatialAttention(nn.Module):
            def __init__(self, kernel_size=3):
                super(SpatialAttention, self).__init__()
                assert kernel_size in (3, 7), "kernel size must be 3 or 7"
                padding = 3 if kernel_size == 7 else 1
                self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False).cuda()

                self.lif = nn.Sequential(
                    neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0,
                                   surrogate_function=surrogate.ATan()),
                )

            def forward(self, x):
                x = rearrange(x, "b f c h w -> b (f c) h w")
                avgout = torch.mean(x, dim=1, keepdim=True)
                maxout, _ = torch.max(x, dim=1, keepdim=True)
                x = torch.cat([avgout, maxout], dim=1)
                x = self.conv(x)
                x = x.unsqueeze(1)
                return self.lif(x)

        self.ta = TimeAttention(T, reduction_t)
        # self.ca = ChannelAttention(C, reduction_c)
        # self.sa = SpatialAttention(kernel_size)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        """
        In the case of mixed accuracy, when jumping from training to testing, the type of reset is float16, and the type of x is float32
        :param x: net input
        """
        assert x.dim() == 5, ValueError(
            f'expected 5D input with shape [T, N, C, H, W], but got input with shape {x.shape}')
        x = x.transpose(0, 1)
        ta = self.ta(x)
        # ca = self.ca(x)
        # sa = self.sa(x)
        out = ta * x
        # out = ca * x
        # out = sa * x
        out = self.relu(out)
        out = out.transpose(0, 1)
        return out


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.first_block = nn.Sequential(
            layer.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            layer.BatchNorm2d(64),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0,
                           surrogate_function=surrogate.ATan()),
        )

        self.block1 = nn.Sequential(
            layer.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            layer.BatchNorm2d(64),
            # MA(16, 64, 4, 8, 3),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0,
                           surrogate_function=surrogate.ATan()),

            layer.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            layer.BatchNorm2d(64),
            MA(16, 64, 4, 8, 3),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0,
                           surrogate_function=surrogate.ATan()),

            layer.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            layer.BatchNorm2d(64),
            # MA(16, 64, 4, 8, 3),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0,
                           surrogate_function=surrogate.ATan()),

            layer.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            layer.BatchNorm2d(64),
            MA(16, 64, 4, 8, 3),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0,
                           surrogate_function=surrogate.ATan()),
        )

        self.block2_1 = nn.Sequential(
            layer.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            layer.BatchNorm2d(128),
            # MA(16, 128, 4, 8, 3),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0,
                           surrogate_function=surrogate.ATan()),

            layer.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            layer.BatchNorm2d(128),
            MA(16, 128, 4, 16, 3),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0,
                           surrogate_function=surrogate.ATan()),
        )

        self.sample1 = nn.Sequential(
            layer.Conv2d(64, 128, kernel_size=1, stride=2, padding=0, bias=False),
            layer.BatchNorm2d(128),
            IA(16, 128, 4, 16, 3),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0,
                           surrogate_function=surrogate.ATan()),
        )

        self.block2_2 = nn.Sequential(
            layer.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            layer.BatchNorm2d(128),
            # MA(16, 128, 4, 16, 3),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0,
                           surrogate_function=surrogate.ATan()),

            layer.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            layer.BatchNorm2d(128),
            MA(16, 128, 4, 16, 3),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0,
                           surrogate_function=surrogate.ATan()),
        )

        self.block3_1 = nn.Sequential(
            layer.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            layer.BatchNorm2d(256),
            # MA(16, 256, 4, 32, 3),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0,
                           surrogate_function=surrogate.ATan()),

            layer.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            layer.BatchNorm2d(256),
            MA(16, 256, 4, 32, 3),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0,
                           surrogate_function=surrogate.ATan()),
        )

        self.sample2 = nn.Sequential(
            layer.Conv2d(128, 256, kernel_size=1, stride=2, padding=0, bias=False),
            layer.BatchNorm2d(256),
            IA(16, 256, 4, 32, 3),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0,
                           surrogate_function=surrogate.ATan()),
        )

        self.block3_2 = nn.Sequential(
            layer.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            layer.BatchNorm2d(256),
            # MA(16, 256, 4, 32, 3),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0,
                           surrogate_function=surrogate.ATan()),

            layer.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            layer.BatchNorm2d(256),
            MA(16, 256, 4, 32, 3),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0,
                           surrogate_function=surrogate.ATan()),
        )

        self.block4_1 = nn.Sequential(
            layer.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
            layer.BatchNorm2d(512),
            # MA(16, 512, 4, 64, 3),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0,
                           surrogate_function=surrogate.ATan()),

            layer.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            layer.BatchNorm2d(512),
            MA(16, 512, 4, 64, 3),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0,
                           surrogate_function=surrogate.ATan()),
        )

        self.sample3 = nn.Sequential(
            layer.Conv2d(256, 512, kernel_size=1, stride=2, padding=0, bias=False),
            layer.BatchNorm2d(512),
            IA(16, 512, 4, 64, 3),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0,
                           surrogate_function=surrogate.ATan()),
        )

        self.block4_2 = nn.Sequential(
            layer.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            layer.BatchNorm2d(512),
            # MA(16, 512, 4, 64, 3),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0,
                           surrogate_function=surrogate.ATan()),

            layer.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            layer.BatchNorm2d(512),
            MA(16, 512, 4, 64, 3),
            neuron.LIFNode(v_threshold=1.0, v_reset=0.0, detach_reset=True, tau=2.0,
                           surrogate_function=surrogate.ATan()),
        )

        self.end_block = nn.Sequential(
            layer.AdaptiveAvgPool2d(1),
            layer.Flatten(start_dim=1, end_dim=-1),
            layer.Linear(512, 10, bias=True)
        )

    def forward(self, x):
        result1 = self.first_block(x)
        result2 = self.block1(result1)
        result3 = (self.block2_1(result2) + self.sample1(result2)) - (self.block2_1(result2) * self.sample1(result2))
        result4 = self.block2_2(result3)
        result5 = (self.block3_1(result4) + self.sample2(result4)) - (self.block3_1(result4) * self.sample2(result4))
        result6 = self.block3_2(result5)
        result7 = (self.block4_1(result6) + self.sample3(result6)) - (self.block4_1(result6) * self.sample3(result6))
        result8 = self.block4_2(result7)
        result = self.end_block(result8)

        return result

Net = ResNet18()

functional.set_step_mode(Net, step_mode='m')
Net.cuda()

parser = argparse.ArgumentParser(description='spikingjelly LIF CIFAR10 Training')

parser.add_argument('--device', default='cuda:0', help='Running equipment\n')
parser.add_argument('--model-output-dir', default='./result_data', help='Path for saving models and results, such as ./  \n')
parser.add_argument('--log-dir', default='./runs', help='Location to save Tensorboard log files\n')
parser.add_argument('--num-workers', default=70, type=int, help='Number of cores used to load the dataset\n')
parser.add_argument('-b', '--batch-size', default=128, type=int, help='Batch\n')
parser.add_argument('-T', '--timesteps', default=16, type=int, dest='T', help='Simulation duration, such as 100 \n')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, metavar='LR', help='Learning rate, such as "1e-3"\n', dest='lr')
parser.add_argument('-N', '--epoch', default=250, type=int, help='Training epoch\n')
parser.add_argument('--decay', default=250, type=int, help='Number of learning rate decay times\n')
parser.add_argument('--local-rank', default=-1, type=int, help='Multi card training, indicating the number of graphics cards (processes)')

def main():
    ''' Conduct one test per epoch '''
    args = parser.parse_args()
    print("############## Parameter details ##############")
    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))  # Output all parameters
    print("###############################################")

    filename = (os.path.basename(__file__))[0:-3]
    log_dir = args.log_dir
    model_output_dir = args.model_output_dir
    batch_size = args.batch_size
    num_steps = args.T
    lr = args.lr
    epochs = args.epoch
    decay = args.decay
    local_rank = args.local_rank            # Number of GPUs

    # DDP：DDP Backend initialization
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')  # Nccl is the fastest and most recommended backend on GPU devices

    writer = SummaryWriter(log_dir)  # For Tensorboard

    max_test_accuracy = 0  # Record the highest test accuracy
    loss_list = []
    train_accs = []
    test_accs = []  # Record test accuracy
    spiking_rate = []  # Used to draw a histogram of the firing frequency of neurons

    scaler = amp.GradScaler()
    functional.set_backend(Net, 'cupy', instance=neuron.LIFNode)

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])

    train_dataset = datasets.CIFAR10('../../data/cifar10', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10('../../data/cifar10', train=False, download=True, transform=transform_test)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)

    train_loader = DataLoader(train_dataset, batch_size, sampler=train_sampler, num_workers=args.num_workers, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size, sampler=test_sampler, num_workers=args.num_workers, drop_last=True)

    print(f"Number of training set samples:{len(train_dataset)}, Number of test set samples:{len(test_dataset)}")

    net = Net
    print(net)
    # build model
    # DDP: The Load model needs to be loaded on the master before constructing the DDP model.
    net = net.to(local_rank)
    net = DDP(net, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # Regularization of learning rate
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=decay)

    def cal_firing_rate(s_seq):
        return s_seq.flatten(0).mean(0)

    # Monitor for recording Spiking rate
    if dist.get_rank() == 0:
        spike_seq_monitor = monitor.OutputMonitor(net, neuron.LIFNode, cal_firing_rate)

    for epoch in range(1, epochs + 1):
        # DDP: Set the epoch of the sampler,
        # DistributedSampler requires this to specify the shuffle method,
        # By maintaining the same random number seed between different processes, different processes can achieve the same shuffle effect.
        train_loader.sampler.set_epoch(epoch)
        test_loader.sampler.set_epoch(epoch)
        train_correct_sum_in_epoch = 0        # The correct number of predictions in the data used for training for each epoch
        train_data_sum_in_epoch = 0           # The total amount of data used for training per epoch
        print(f"train epoch : {epoch}")
        net.train()
        if dist.get_rank() == 0:
            spike_seq_monitor.clear_recorded_data()     # Clear the recorded Spiking rate
            spiking_rate.clear()                        # Clear the list of spike rates in the record
            spike_seq_monitor.disable()         # Stop recording during training phase
        for img, label in tqdm(train_loader):
            img = img.to(local_rank)
            img = img.unsqueeze(0).repeat(num_steps, 1, 1, 1, 1)
            label = label.to(local_rank)
            label_one_hot = F.one_hot(label, 10).float()            # Encode the label one hot for later loss calculation
            # Mixed precision training
            with amp.autocast():
                result = net(img).mean(0)
                # Calculate the correct quantity for classification
                pred = result.argmax(dim=1)
                correct = pred.eq(label).sum().float().item()  # Correct quantity (float)
                train_correct_sum_in_epoch += correct  # Record the correct quantity for each epoch
                loss = F.mse_loss(result, label_one_hot)  # MSE
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            functional.reset_net(net)  # Reset network status
            train_data_sum_in_epoch += label.numel()  # The numel function returns the number of elements in an array, with a value of the total number of images
        # Add accurate quantity and total quantity across threads
        train_correct_sum_in_epoch = torch.tensor(train_correct_sum_in_epoch).to(local_rank)
        train_data_sum_in_epoch = torch.tensor(train_data_sum_in_epoch).to(local_rank)
        torch.distributed.all_reduce(train_correct_sum_in_epoch, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(train_data_sum_in_epoch, op=torch.distributed.ReduceOp.SUM)
        if dist.get_rank() == 0:
            loss_list.append(loss.item())
            train_accuracy_in_epoch = train_correct_sum_in_epoch / train_data_sum_in_epoch  # The accuracy of each epoch
            train_accs.append(train_accuracy_in_epoch)
            writer.add_scalars('train_and_test_accuracy', {'train': train_accuracy_in_epoch}, epoch)
            writer.add_scalar('loss_epoch', loss.item(), epoch)
        lr_scheduler.step()
        print("###############   End of training, start testing   ###############")
        net.eval()
        if dist.get_rank() == 0:
            spike_seq_monitor.enable()      # Start recording Spiking rate
        with torch.no_grad():  # Conduct one test per epoch
            test_data_correct_sum = 0  # The correct number of model outputs during the testing process
            test_data_sum = 0  # Total data volume during the testing process
            for img, label in tqdm(test_loader):
                img = img.to(local_rank)
                img = img.unsqueeze(0).repeat(num_steps, 1, 1, 1, 1)
                label = label.to(local_rank)
                result = net(img).mean(0)
                # Calculate the correct quantity for classification
                pred = result.argmax(dim=1)
                correct = pred.eq(label).sum().float().item()
                test_data_correct_sum += correct
                test_data_sum += label.numel()
                functional.reset_net(net)
            # Add accurate quantity and total quantity across threads
            test_data_correct_sum = torch.tensor(test_data_correct_sum).to(local_rank)
            test_data_sum = torch.tensor(test_data_sum).to(local_rank)
            torch.distributed.all_reduce(test_data_correct_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(test_data_sum, op=torch.distributed.ReduceOp.SUM)
        if dist.get_rank() == 0:
            test_accuracy = test_data_correct_sum / test_data_sum  # Test accuracy
            writer.add_scalars('train_and_test_accuracy', {'test': test_accuracy}, epoch)
            test_accs.append(test_accuracy)
            # Determine if this result is currently the best result
            save_max = False
            if test_accuracy > max_test_accuracy:
                max_test_accuracy = test_accuracy
                save_max = True
            print(f"epoch{epoch}中，The accuracy of the training set is{train_accuracy_in_epoch}, The testing accuracy "
                  f"is{test_accuracy},The best testing accuracy to date is{max_test_accuracy}")
            writer.close()
            # It seems that saving the model cannot coexist with the hook created when saving the spike rate
            # if save_max:
            #     torch.save(net, os.path.join(model_output_dir, filename + '_max.pt'))
            if not os.path.exists(model_output_dir):
                os.makedirs(model_output_dir)
            if not os.path.exists(model_output_dir + '/spiking_rate/' + filename):
                os.makedirs(model_output_dir + '/spiking_rate/' + filename)
            # Save location list only in the last epoch
            if epoch == epochs:
                torch.save(spike_seq_monitor.monitored_layers, model_output_dir + '/spiking_rate/' +
                           filename + '/' + 'name_list.pt')

            # Organize spike rate format
            for i in range(len(spike_seq_monitor.monitored_layers)):
                spiking_rate.append(spike_seq_monitor[spike_seq_monitor.monitored_layers[i]][0])

            if save_max == True:
                # saving firinging rate
                torch.save(spiking_rate, model_output_dir + '/spiking_rate/' +
                               filename + '/' + 'spikingrate_' + 'epoch' + str(epoch) + '.pt')

    if dist.get_rank() == 0:
        train_accs = np.array(torch.tensor(train_accs, device='cpu'))
        np.save(model_output_dir + '/' + filename + '_train_acc.npy', train_accs)
        test_accs = np.array(torch.tensor(test_accs, device='cpu'))
        np.save(model_output_dir + '/' + filename + '_test_acc.npy', test_accs)
        loss_list = np.array(torch.tensor(loss_list, device='cpu'))
        np.save(model_output_dir + '/' + filename + '_loss.npy', loss_list)


if __name__ == '__main__':
    main()


