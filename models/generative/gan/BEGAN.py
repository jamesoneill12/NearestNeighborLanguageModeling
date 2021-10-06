from ..utils import *
import torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from ..dataloader import dataloader
from evaluators.gan.metrics import compute_score, inception_score
from models.distributions.mix_distribution import mix_dist
from models.samplers.ss import update_ss_prob
import pickle

class generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=100, output_dim=1, input_size=32):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )
        initialize_weights(self)

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv(x)

        return x

class discriminator(nn.Module):
    # It must be Auto-Encoder style architecture
    # Architecture : (64)4c2s-FC32-FC64*14*14_BR-(1)4dc2s_S
    def __init__(self, input_dim=1, output_dim=1, input_size=32):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * (self.input_size // 2) * (self.input_size // 2), 32),
            nn.Linear(32, 64 * (self.input_size // 2) * (self.input_size // 2)),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            #nn.Sigmoid(),
        )

        initialize_weights(self)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        x = x.view(-1, 64, (self.input_size // 2), (self.input_size // 2))
        x = self.deconv(x)

        return x

class BEGAN(object):
    def __init__(self, args):
        # parameters
        self.epoch = args.epoch
        self.sample_num = args.sample_num
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.gpu_mode = args.gpu_mode
        self.model_name = args.gan_type
        self.input_size = args.input_size
        self.z_dim = args.z_dim # 62
        self.gamma = 1
        self.lambda_ = 0.001
        self.k = 0.0
        self.lr_lower_boundary = 0.00002

        self.mixture = args.mixture
        self.mix_noise = args.mix_noise
        self.mix_direction = args.mix_direction
        "0: discriminator keeps transferred features"
        "1: discriminator loses transferred features"
        "2: discriminator and generator features are swapped"
        self.mix_lb = args.mix_lb
        self.mix_ub = args.mix_ub

        self.use_tensorboard = args.use_tensorboard
        self.image_path = args.data
        self.sample_path = args.sample_dir
        self.log_step = args.log_step
        self.sample_step = args.sample_step
        self.model_save_step = args.model_save_step
        self.results_save_step = args.results_save_step
        self.version = args.version

        self.results_path = os.path.join(args.result_dir, self.version)
        self.log_path = os.path.join(args.log_dir, self.version)
        self.sample_path = os.path.join(args.sample_dir, self.version)

        if not os.path.exists(self.sample_path):
            os.makedirs(self.sample_path)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        if os.path.isfile(self.results_path + '.pkl'):
            with open(self.results_path + '.pkl', 'rb') as f:
                self.results = pickle.load(f)
        else:
            self.results = [] # {'kid': [], 'fid': [], 'is': []}


        # load dataset
        self.data_loader = dataloader(self.dataset, self.input_size, self.batch_size)
        data = self.data_loader.__iter__().__next__()[0]

        # networks init
        self.G = generator(input_dim=self.z_dim, output_dim=data.shape[1], input_size=self.input_size)
        self.D = discriminator(input_dim=data.shape[1], output_dim=1, input_size=self.input_size)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=0.0002, betas=(args.beta1, args.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=0.0002, betas=(args.beta1, args.beta2))

        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()

        print('---------- Networks architecture -------------')
        print_network(self.G)
        print_network(self.D)
        print('-----------------------------------------------')

        # fixed noise
        self.sample_z_ = torch.rand((self.batch_size, self.z_dim))
        if self.gpu_mode:
            self.sample_z_ = self.sample_z_.cuda()

    def train(self):

        self.train_hist = {}
        all_fake, all_real = [], []
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []
        self.M = {}
        self.M['pre'] = []
        self.M['pre'].append(1)
        self.M['cur'] = []

        self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)
        if self.gpu_mode:
            self.y_real_, self.y_fake_ = self.y_real_.cuda(), self.y_fake_.cuda()

        self.D.train()
        print('training start!!')
        start_time = time.time()
        for epoch in range(self.epoch):
            self.G.train()
            epoch_start_time = time.time()
            for iter, (x_, _) in enumerate(self.data_loader):
                if iter == self.data_loader.dataset.__len__() // self.batch_size:
                    break

                z_ = torch.rand((self.batch_size, self.z_dim))

                if self.gpu_mode:
                    x_, z_ = x_.cuda(), z_.cuda()

                # update D network
                self.D_optimizer.zero_grad()

                D_real = self.D(x_)
                D_real_loss = torch.mean(torch.abs(D_real - x_))

                G_ = self.G(z_)

                D_fake = self.D(G_)
                D_fake_loss = torch.mean(torch.abs(D_fake - G_))

                D_loss = D_real_loss - self.k * D_fake_loss
                self.train_hist['D_loss'].append(D_loss.item())

                D_loss.backward()
                self.D_optimizer.step()

                # update G network
                self.G_optimizer.zero_grad()

                G_ = self.G(z_)

                if self.mixture is not None:
                    if self.mixture == 'static':
                        rate = self.mix_ub
                    else:
                        rate = self.mix_ub * (1-epoch/self.epoch)
                        rate = update_ss_prob(rate,  decay=self.mixture, uthresh=self.mix_ub)
                    x_, G_ = mix_dist(x_, G_, self.mixture, p=rate, disc_rep=False, eps=self.mix_noise)

                D_fake = self.D(G_)
                D_fake_loss = torch.mean(torch.abs(D_fake - G_))

                G_loss = D_fake_loss
                self.train_hist['G_loss'].append(G_loss.item())

                G_loss.backward()
                self.G_optimizer.step()

                # print(G_.size()) # checking for when concat to compute i_s
                if epoch+1 == self.epoch:
                    all_fake.append(G_.cpu().data)
                    all_real.append(x_.cpu().data)

                # convergence metric
                temp_M = D_real_loss + torch.abs(self.gamma * D_real_loss - G_loss)

                # operation for updating k
                temp_k = self.k + self.lambda_ * (self.gamma * D_real_loss - G_loss)
                temp_k = temp_k.item()

                self.k = min(max(temp_k, 0), 1)
                self.M['cur'] = temp_M.item()

                if ((iter + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size, D_loss.item(), G_loss.item()))

                # compute frechet distance and kernel polar distance
                if ((iter + 1) % self.results_save_step) == 0:
                    # self.subset_size
                    cs = compute_score(x_.data.cpu(), G_.data.cpu())
                    print("fid: {}\t mmd: {}\t emd: {}\t knn: {} \t inception: {}".
                          format(cs.fid, cs.mmd, cs.emd, cs.knn.acc, cs.i_s))
                    self.results.append(cs)

            # if epoch == 0:
            #     self.M['pre'] = self.M['cur']
            #     self.M['cur'] = []
            # else:
            if np.mean(self.M['pre']) < np.mean(self.M['cur']):
                pre_lr = self.G_optimizer.param_groups[0]['lr']
                self.G_optimizer.param_groups[0]['lr'] = max(self.G_optimizer.param_groups[0]['lr'] / 2.0,
                                                             self.lr_lower_boundary)
                self.D_optimizer.param_groups[0]['lr'] = max(self.D_optimizer.param_groups[0]['lr'] / 2.0,
                                                             self.lr_lower_boundary)
                print('M_pre: ' + str(np.mean(self.M['pre'])) + ', M_cur: ' + str(
                    np.mean(self.M['cur'])) + ', lr: ' + str(pre_lr) + ' --> ' + str(
                    self.G_optimizer.param_groups[0]['lr']))
            else:
                print('M_pre: ' + str(np.mean(self.M['pre'])) + ', M_cur: ' + str(np.mean(self.M['cur'])))
                self.M['pre'] = self.M['cur']

                self.M['cur'] = []

            self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
            with torch.no_grad():
                self.visualize_results((epoch+1))

        all_fake = torch.cat(all_fake, 0)
        all_real = torch.cat(all_real, 0)
        perm = torch.randperm(all_fake.size(0))
        idx = perm[:self.sample_num]
        all_fake, all_real = all_fake[idx], all_real[idx]

        if all_fake.size(1) < 2:
            all_fake = torch.cat([all_fake]*3, 1)
            all_real = torch.cat([all_real] * 3, 1)
        # just do it at the end, too expensive during training
        # couldn't fit in memory so changed cuda=True to False

        cs = compute_score(all_real, all_fake, inception=True, cuda=True, bsize=32)
        # if self.dataset == 'mnist'
        self.results.append(cs)
        del all_fake, all_real

        with open(self.results_path + '.pkl', 'wb') as f:
            pickle.dump(self.results, f)

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
              self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save()
        generate_animation(self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name,
                                 self.epoch)
        loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.model_name), self.model_name)

    def visualize_results(self, epoch, fix=True):
        self.G.eval()

        if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name):
            os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name)

        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        if fix:
            """ fixed noise """
            samples = self.G(self.sample_z_)
        else:
            """ random noise """
            sample_z_ = torch.rand((self.batch_size, self.z_dim))
            if self.gpu_mode:
                sample_z_ = sample_z_.cuda()

            samples = self.G(sample_z_)

        if self.gpu_mode:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        samples = (samples + 1) / 2
        save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          self.result_dir + '/' + self.dataset + '/' + self.model_name + '/' + self.model_name + '_epoch%03d' % epoch + '.png')

    def save(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))

        with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))