""" Self-Attention Generative Adversarial Network """

from models.networks.generative.gan.spectral import SpectralNorm
from ..dataloader import dataloader
import time
import datetime
from torchvision.utils import save_image
from models.networks.generative.utils import *
from models.distributions.mix_distribution import mix_dist
from models.samplers.ss import update_ss_prob
from evaluators.gan.kernel import polynomial_mmd_averages
from evaluators.gan.frechet import calculate_frechet_distance
from evaluators.gan.metrics import compute_score, inception_score
import pickle


class Self_Attn(nn.Module):
    """ Self attention Layer """
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #

    def forward(self,x):
        """ inputs:
        x: input
        feature
        maps(B X C X W X H)
        returns:
        out: self
        attention
        value + input
        feature attention: B X N X
        N(N is Width * Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)

        out = self.gamma*out + x
        return out,attention


class Generator(nn.Module):
    """ Generator. """

    def __init__(self, batch_size, image_size=64, z_dim=100, conv_dim=64, n_channels = 3):
        super(Generator, self).__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        repeat_num = int(np.log2(self.imsize)) - 3
        mult = 2 ** repeat_num # 8
        layer1.append(SpectralNorm(nn.ConvTranspose2d(z_dim, conv_dim * mult, 4)))
        layer1.append(nn.BatchNorm2d(conv_dim * mult))
        layer1.append(nn.ReLU())

        curr_dim = conv_dim * mult

        layer2.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer2.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer2.append(nn.ReLU())

        curr_dim = int(curr_dim / 2)

        layer3.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer3.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer3.append(nn.ReLU())

        layer4 = []
        curr_dim = int(curr_dim / 2)
        layer4.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer4.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer4.append(nn.ReLU())
        self.l4 = nn.Sequential(*layer4)
        curr_dim = int(curr_dim / 2)

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.ConvTranspose2d(curr_dim, n_channels, 4, 2, 1))
        last.append(nn.Tanh())
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn( 128, 'relu')
        self.attn2 = Self_Attn( 64,  'relu')

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        out=self.l1(z)
        out=self.l2(out)
        out=self.l3(out)
        out,p1 = self.attn1(out)
        out=self.l4(out)
        out,p2 = self.attn2(out)
        out=self.last(out)

        return out, p1, p2


class Discriminator(nn.Module):
    """ Discriminator, Auxiliary Classifier. """

    def __init__(self, batch_size=64, image_size=64, conv_dim=64, n_channels = 3):
        super(Discriminator, self).__init__()
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        layer1.append(SpectralNorm(nn.Conv2d(n_channels, conv_dim, 4, 2, 1)))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        print("{} imsize ".format(self.imsize))

        layer4 = []
        layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer4.append(nn.LeakyReLU(0.1))
        self.l4 = nn.Sequential(*layer4)
        curr_dim = curr_dim*2

        last.append(nn.Conv2d(curr_dim, 2, 2))
        self.last = nn.Sequential(*last)

        self.attn1 = Self_Attn(256, 'relu')
        self.attn2 = Self_Attn(512, 'relu')

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out,p1 = self.attn1(out)
        out=self.l4(out)
        out,p2 = self.attn2(out)
        # print(out.size())
        out=self.last(out)
        return out.squeeze(), p1, p2


class SAGAN(object):
    def __init__(self, config):

        self.save_dir = config.save_dir
        self.result_dir = config.result_dir
        self.dataset = config.dataset
        self.log_path = config.log_dir
        self.log_dir = config.log_dir
        self.input_size = config.input_size
        self.epoch = config.epoch
        self.sample_num = config.sample_num
        self.batch_size = config.batch_size
        self.subset_size = config.subset_size
        self.gpu_mode = config.gpu_mode
        self.model_name = config.gan_type
        self.z_dim = config.z_dim
        self.use_tensorboard = config.use_tensorboard

        # Data loader
        self.data_loader = dataloader(self.dataset, self.input_size, self.batch_size)
        self.n_channels = 1 if 'mnist' in self.dataset else 3
        self.mixture = config.mixture
        self.mix_noise = config.mix_noise
        self.mix_direction = config.mix_direction
        "0: discriminator keeps transferred features"
        "1: discriminator loses transferred features"
        "2: discriminator and generator features are swapped"
        self.mix_lb = config.mix_lb
        self.mix_ub = config.mix_ub

        # exact model and loss
        self.model = config.model
        self.adv_loss = config.adv_loss

        # Model hyper-parameters
        self.imsize = config.imsize
        self.g_num = config.g_num
        self.z_dim = config.z_dim
        self.g_conv_dim = config.g_conv_dim
        self.d_conv_dim = config.d_conv_dim
        self.parallel = config.parallel

        self.lambda_gp = config.lambda_gp
        self.total_step = config.total_step
        self.d_iters = config.d_iters
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.lr_decay = config.lr_decay
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.pretrained_model = config.pretrained_model

        self.dataset = config.dataset
        self.use_tensorboard = config.use_tensorboard
        self.image_path = config.data
        self.save_dir = config.save_dir
        self.sample_path = config.sample_dir

        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step
        self.results_save_step = config.results_save_step
        self.version = config.version

        # Path
        self.results_path = os.path.join(config.result_dir, self.version)
        self.log_path = os.path.join(config.log_dir, self.version)
        self.sample_path = os.path.join(config.sample_dir, self.version)
        self.save_dir = os.path.join(config.save_dir, self.version)

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

        self.build_model()

        if self.use_tensorboard:
            self.build_tensorboard()

        # Start with trained model
        if self.pretrained_model:
            self.load_pretrained_model()

        # fixed noise
        self.sample_z_ = torch.rand((self.batch_size, self.z_dim))
        if self.gpu_mode:
            self.sample_z_ = self.sample_z_.cuda()

    def train(self):

        # Data iterator
        # data_iter = iter(self.data_loader)
        # step_per_epoch = len(self.data_loader)
        # model_save_step = int(self.model_save_step * step_per_epoch)
        # results_save_step = int(self.results_save_step * step_per_epoch)

        # Fixed input for debugging
        # fixed_z = tensor2var(torch.randn(self.batch_size, self.z_dim))

        # Start with trained model
        if self.pretrained_model:
            start = self.pretrained_model + 1
        else:
            start = 0

        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []

        self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)

        if self.gpu_mode:
            self.y_real_, self.y_fake_ = self.y_real_.cuda(), self.y_fake_.cuda()

        self.D.train()
        print('training start!!')
        start_time = time.time()

        all_fake, all_real = [], []

        for epoch in range(self.epoch):
        # step -> epoch
        #for step in range(start, self.total_step):
            # Start time
            epoch_start_time = time.time()
            self.G.train()

            for iter, (x_, _) in enumerate(self.data_loader):

                if iter == self.data_loader.dataset.__len__() // self.batch_size:
                    break
                # ================== Train D ================== #

                """
                try:
                    real_images, _ = next(data_iter)
                except:
                    data_iter = iter(self.data_loader)
                    real_images, _ = next(data_iter)                
                """

                real_images = x_
                #fake_images = torch.rand((self.batch_size, self.z_dim))
                if self.gpu_mode: real_images =  real_images.cuda()

                # print(real_images.size())

                # Compute loss with real images
                # dr1, dr2, df1, df2, gf1, gf2 are attention scores
                real_images = tensor2var(real_images)
                # print("real img {}".format(real_images.size()))
                d_out_real,dr1,dr2 = self.D(real_images)

                if self.adv_loss == 'wgan-gp':
                    d_loss_real = - torch.mean(d_out_real)
                elif self.adv_loss == 'hinge':
                    d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()

                # apply Gumbel Softmax
                z = tensor2var(torch.randn(real_images.size(0), self.z_dim))
                fake_images, gf1, gf2 = self.G(z)
                # print("fake img {}".format(fake_images.size()))
                d_out_fake, df1, df2 = self.D(fake_images)

                if self.adv_loss == 'wgan-gp':
                    d_loss_fake = d_out_fake.mean()
                elif self.adv_loss == 'hinge':
                    d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()

                # Backward + Optimize
                d_loss = d_loss_real + d_loss_fake
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                if self.adv_loss == 'wgan-gp':
                    # Compute gradient penalty
                    alpha = torch.rand(real_images.size(0), 1, 1, 1).cuda().expand_as(real_images)
                    interpolated = Variable(alpha * real_images.data + (1 - alpha) * fake_images.data, requires_grad=True)
                    out,_,_ = self.D(interpolated)

                    grad = torch.autograd.grad(outputs=out,
                                               inputs=interpolated,
                                               grad_outputs=torch.ones(out.size()).cuda(),
                                               retain_graph=True,
                                               create_graph=True,
                                               only_inputs=True)[0]

                    grad = grad.view(grad.size(0), -1)
                    grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
                    d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

                    # Backward + Optimize
                    d_loss = self.lambda_gp * d_loss_gp

                    self.reset_grad()
                    d_loss.backward()
                    self.d_optimizer.step()

                # ================== Train G and gumbel ================== #
                # Create random noise
                z = tensor2var(torch.randn(real_images.size(0), self.z_dim))
                fake_images,_,_ = self.G(z)

                if self.mixture is not None:
                    if self.mixture == 'static':
                        rate = self.mix_ub
                    else:
                        rate = self.mix_ub * (1-epoch/self.epoch)
                        rate = update_ss_prob(rate,  decay=self.mixture, uthresh=self.mix_ub)
                    _, fake_images = mix_dist(real_images, fake_images, self.mixture, p=rate, disc_rep=False, eps=self.mix_noise)

                # Compute loss with fake images
                g_out_fake,_,_ = self.D(fake_images)  # batch x n
                if self.adv_loss == 'wgan-gp':
                    g_loss_fake = - g_out_fake.mean()
                elif self.adv_loss == 'hinge':
                    g_loss_fake = - g_out_fake.mean()

                self.reset_grad()
                g_loss_fake.backward()
                self.g_optimizer.step()

                # print(G_.size()) # checking for when concat to compute i_s
                if epoch+1 == self.epoch:
                    all_fake.append(fake_images.cpu().data)
                    all_real.append(real_images.cpu().data)


                # Print out log info
                #if (step + 1) % self.log_step == 0:
                if ((iter + 1) % 100) == 0:
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))

                    print("Elapsed [{}], epoch [{}] G_step [{}/{}], D_step[{}/{}], d_out_real: {:.4f}, "
                          " ave_gamma_l3: {:.4f}, ave_gamma_l4: {:.4f}".
                          format(elapsed, (epoch+1), (iter + 1), self.data_loader.dataset.__len__() // self.batch_size, (iter + 1),
                                 self.data_loader.dataset.__len__() // self.batch_size , d_loss_real.item(),
                                 self.G.attn1.gamma.mean().item(), self.G.attn2.gamma.mean().item() ))

                # compute frechet distance and kernel inception distance
                if ((iter + 1) % self.results_save_step) == 0:
                    # self.subset_size
                    cs = compute_score(real_images.data.cpu(), fake_images.data.cpu())
                    print("fid: {}\t mmd: {}\t emd: {}\t knn: {} \t inception: {}".
                          format(cs.fid, cs.mmd, cs.emd, cs.knn.acc, cs.i_s))
                    self.results.append(cs)

                # Sample images
                if (iter + 1) % self.sample_step == 0:
                    fake_images,_,_= self.G(z)
                    save_image(denorm(fake_images.data),
                               os.path.join(self.sample_path, '{}_fake.png'.format(iter + 1)))

                """
                if (iter+1) % model_save_step==0:
                    torch.save(self.G.state_dict(),
                               os.path.join(self.save_dir, '{}_G.pth'.format(iter + 1)))
                    torch.save(self.D.state_dict(),
                               os.path.join(self.save_dir, '{}_D.pth'.format(iter + 1)))                

                # compute frechet distance and kernel polar distance
                if (iter+1) % results_save_step ==0:
                    fake_images,_,_= self.G(fixed_z)
                    # self.subset_size
                    kid = polynomial_mmd_averages(fake_images, real_images, subset_size=self.batch_size,ret_var=True)
                    #, output=sys.stdout, **kernel_args)
                    mu_f, sigma_f = fake_images.mean([3, 4]), fake_images.std(dim=[3, 4])
                    mu_r, sigma_r = real_images.mean([3, 4]), real_images.std(dim=[3, 4])
                    fid = calculate_frechet_distance(mu_f, sigma_f, mu_r, sigma_r, eps=1e-6)
                    i_s = None
                    self.results['kid'].append((step, kid))
                    self.results['fid'].append((step, fid))
                    self.results['is'].append((step, i_s))

                """

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
            # samples = tensor2var(torch.randn(self.batch_size, self.z_dim))
            samples, _, _ = self.G(self.sample_z_)
        else:
            """ random noise """
            sample_z_ = torch.rand((self.batch_size, self.z_dim))
            if self.gpu_mode: sample_z_ = sample_z_.cuda()
            samples, _, _ = self.G(sample_z_)

        # print(samples.size())
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


    def build_model(self):

        data = self.data_loader.__iter__().__next__()[0]
        # not using data.shape[1] for output dim

        self.G = Generator(self.batch_size, self.imsize, self.z_dim, self.g_conv_dim, n_channels=self.n_channels).cuda()
        self.D = Discriminator(self.batch_size, self.imsize, self.d_conv_dim, n_channels=self.n_channels).cuda()
        if self.parallel:
            self.G = nn.DataParallel(self.G)
            self.D = nn.DataParallel(self.D)

        # Loss and optimizer
        # self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.g_lr, [self.beta1, self.beta2])
        self.g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), self.g_lr, [self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), self.d_lr, [self.beta1, self.beta2])

        self.c_loss = torch.nn.CrossEntropyLoss()
        # print networks
        print(self.G)
        print(self.D)


    def build_tensorboard(self):
        from tensorboardX import SummaryWriter
        self.logger = SummaryWriter(self.log_path)

    def load_pretrained_model(self):
        self.G.load_state_dict(torch.load(os.path.join(
            self.save_dir, '{}_G.pth'.format(self.pretrained_model))))
        self.D.load_state_dict(torch.load(os.path.join(
            self.save_dir, '{}_D.pth'.format(self.pretrained_model))))
        print('loaded trained mods (step: {})..!'.format(self.pretrained_model))

    def reset_grad(self):
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()

    def save_sample(self, data_iter):

        real_images, _ = next(data_iter)
        save_image(denorm(real_images), os.path.join(self.sample_path, 'real.png'))