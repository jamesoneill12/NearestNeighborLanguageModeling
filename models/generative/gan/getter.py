def get_gan(args):

    which = args.gan_type.lower()
    if which == 'gan': from models.networks.generative.gan.GAN import GAN; mod = GAN
    elif which == 'began': from models.networks.generative.gan.BEGAN import BEGAN; mod = BEGAN
    elif which == 'cgan': from models.networks.generative.gan.CGAN import CGAN; mod = CGAN
    elif which == 'acgan': from models.networks.generative.gan.ACGAN import ACGAN; mod = ACGAN
    elif which == 'dragan': from models.networks.generative.gan.DRAGAN import DRAGAN; mod = DRAGAN
    elif which == 'biggan': from models.networks.generative.gan.BigGAN import BigGAN; mod = BigGAN
    elif which == 'ebgan': from models.networks.generative.gan.EBGAN import EBGAN; mod = EBGAN
    elif which == 'infogan': from models.networks.generative.gan.infoGAN import infoGAN; mod = infoGAN
    elif which == 'lsgan': from models.networks.generative.gan.LSGAN import LSGAN; mod = LSGAN
    elif which == 'rsgan': from models.networks.generative.gan.RSGAN import RSGAN; mod = RSGAN
    elif which == 'wgan': from models.networks.generative.gan.WGAN import WGAN; mod = WGAN
    elif which == 'wgan_gp': from models.networks.generative.gan.WGAN_GP import WGAN_GP; mod = WGAN_GP
    elif which == 'sagan': from models.networks.generative.gan.SAGAN import SAGAN; mod = SAGAN
    elif which == 'reinforcegan': from models.networks.generative.gan.ReinforceGAN import ReinforceGAN; mod = ReinforceGAN
    elif which == 'simgan': from models.networks.generative.gan.SimGAN import SimGAN; mod = SimGAN
    else: raise Exception("[!] There is no option for " + args.gan_type)
    gan = mod(args)
    return gan


if __name__ == "__main__":

    pass



