from matplotlib.pyplot import axes
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as dist

### Follows model as seen in LEARNING ROBUST REPRESENTATIONS BY PROJECTING SUPERFICIAL STATISTICS OUT

# Calculate the gaussianity
def nogauss(a):
    num = a.shape[1]
    std = torch.std(a, dim=1, keepdim=True).repeat(1,num)
    mean = torch.mean(a, dim=1, keepdim=True).repeat(1,num)
    cal = (a-mean)/std
    y = torch.mean(torch.pow(cal,4),1)-3*torch.pow(torch.mean(torch.pow(cal,2),1),2)
    return torch.mean(torch.abs(y))

def compute_kernel(x, y):
    x_size = x.shape[0]
    y_size = y.shape[0]
    dim = x.shape[1]

    tiled_x = x.view(x_size,1,dim).repeat(1, y_size,1)
    tiled_y = y.view(1,y_size,dim).repeat(x_size, 1,1)

    return torch.exp(-torch.mean((tiled_x - tiled_y)**2,dim=2)/dim*1.0)

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return torch.mean(x_kernel) + torch.mean(y_kernel) - 2*torch.mean(xy_kernel)

# Encoders
class qzd(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(qzd, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        # self.encoder = enconder_d

        self.fc11 = nn.Sequential(nn.Linear(1024, 256), nn.ReLU())
        self.fc12 = nn.Sequential(nn.Linear(256, zd_dim))
        self.fc21 = nn.Sequential(nn.Linear(1024, 256))
        self.fc22 = nn.Sequential(nn.Linear(256, zd_dim), nn.ReLU())
        self.d_likelihood = nn.Linear(zd_dim, d_dim)

        torch.nn.init.xavier_uniform_(self.encoder[0].weight)
        torch.nn.init.xavier_uniform_(self.encoder[4].weight)
        torch.nn.init.xavier_uniform_(self.fc11[0].weight)
        self.fc11[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc12[0].weight)
        self.fc12[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc21[0].weight)
        self.fc21[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc22[0].weight)
        self.fc22[0].bias.data.zero_()

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(-1, 1024)
        zd_mu = self.fc11(h)
        zd_mu = self.fc12(zd_mu)
        zd_sigma = self.fc21(h)
        zd_sigma = self.fc22(zd_sigma)
        zd_sigma = torch.exp(0.5 * zd_sigma)

        qzd = dist.Normal( zd_mu, zd_sigma)
        zd_q = qzd.rsample()
        logit_d = self.d_likelihood(zd_q)
        return zd_mu, zd_sigma, logit_d


class zd_to_zy(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super( zd_to_zy, self).__init__()


        self.fc21 = nn.Sequential(nn.Dropout(),
                                  nn.Linear(zd_dim, zd_dim * 4),
                                  nn.ReLU(),
                                  nn.Linear(4*zd_dim, zd_dim),
                                  nn.ReLU())
        self.fc22 = nn.Sequential(nn.Dropout(),
                                  nn.Linear(zd_dim, zd_dim * 4),
                                  nn.ReLU(),
                                  nn.Linear(4*zd_dim, zd_dim),
                                  nn.ReLU())


        # torch.nn.init.xavier_uniform_(self.fc21[0].weight)
        # self.fc21[0].bias.data.zero_()
        # torch.nn.init.xavier_uniform_(self.fc22[0].weight)
        # self.fc22[0].bias.data.zero_()

    def forward(self, x_1, x_2):

        zy_mu = self.fc21(x_1)
        zy_sigma = self.fc22(x_2)
        zy_sigma = torch.exp(0.5 * zy_sigma)


        return zy_mu, zy_sigma


class py(nn.Module):
    def __init__(self, d_dim, x_dim, y_dim, zd_dim, zx_dim, zy_dim):
        super(py, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
        )

        self.fc11 = nn.Sequential(nn.Linear(1024, 256), nn.ReLU())
        self.fc12 = nn.Sequential(nn.Linear(256+zd_dim, y_dim))


        torch.nn.init.xavier_uniform_(self.encoder[0].weight)
        torch.nn.init.xavier_uniform_(self.encoder[4].weight)
        torch.nn.init.xavier_uniform_(self.fc11[0].weight)
        self.fc11[0].bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.fc12[0].weight)
        self.fc12[0].bias.data.zero_()


    def forward(self, x, z):
        h = self.encoder(x)
        h = h.view(-1, 1024)
        h = self.fc11(h)
        h_cat_z = torch.cat((h, z), dim=-1)

        y_hat = self.fc12(h_cat_z)
        return h, h_cat_z, y_hat


class JVINet(nn.Module):
    def __init__(self, args):
        super(JVINet, self).__init__()
        self.zd_dim = args.zd_dim
        self.zx_dim = args.zx_dim
        self.zy_dim = args.zy_dim
        self.d_dim = args.d_dim
        self.x_dim = args.x_dim
        self.y_dim = args.y_dim

        self.qzd = qzd(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)
        self.zd_to_zy = zd_to_zy(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)
        self.py = py(self.d_dim, self.x_dim, self.y_dim, self.zd_dim, self.zx_dim, self.zy_dim)

        self.beta_d = args.beta_d
        self.beta_x = args.beta_x
        self.beta_y = args.beta_y
        self.beta_n = args.beta_n
        self.aux_loss_multiplier_d = args.aux_loss_multiplier_d
        self.aux_loss_multiplier_y = args.aux_loss_multiplier_y

        self.cuda()

    def forward(self, x):
        # Encode
        zd_q_loc, zd_q_scale, logit_d = self.qzd(x)

        zy_q_loc, zy_q_scale = self.zd_to_zy(zd_q_loc, zd_q_scale)
        


        # Reparameterization trick
        qzd = dist.Normal(zd_q_loc, zd_q_scale)
        # zd_q = qzd.rsample()


        qzy = dist.Normal(zy_q_loc, zy_q_scale)
        zy_q = qzy.rsample()


        # Decode
        # zdpzxp = torch.cat((zd_p, zx_p), dim=-1)
        d_hat = logit_d

        # zypzxp = torch.cat((zy_p, zx_p), dim=-1)
        _, _, y_hat = self.py(x, zy_q)

        # return x_recon, d_hat, y_hat, qzd, pzd, zd_q, qzx, pzx, zx_q, qzy, pzy, zy_q
        return zd_q_loc, zd_q_scale, zy_q_loc, zy_q_scale, d_hat, y_hat


    def loss_function(self, d, x, y):
        if y is None:  # unsupervised
            pass
        else: # supervised           
            zd_q_loc, zd_q_scale, zy_q_loc, zy_q_scale, d_hat, y_hat = self.forward(x)


            # KLD_y = 0.5*(torch.mean(torch.sum(torch.log(1/zy_q_scale), dim=1), dim=0)-1+\
            #                torch.mean(torch.sum(1/1*zy_q_scale, dim=1), dim=0)+\
            #                torch.mean(torch.sum((zd_q_loc-zy_q_loc)*(1/1)*(zd_q_loc-zy_q_loc), dim=1), dim=0))

            KLD_y = 0.5*(torch.mean(torch.sum(torch.log(zd_q_scale/zy_q_scale), dim=1), dim=0)-1+\
                           torch.mean(torch.sum(1/zd_q_scale*zy_q_scale, dim=1), dim=0)+\
                           torch.mean(torch.sum((zd_q_loc-zy_q_loc)*(1/zd_q_scale)*(zd_q_loc-zy_q_loc), dim=1), dim=0))


            _, d_target = d.max(dim=1)
            CE_d = F.cross_entropy(d_hat, d_target, reduction='sum')

            _, y_target = y.max(dim=1)
            CE_y = F.cross_entropy(y_hat, y_target, reduction='sum')

            # with open('sample_value.txt','a') as t:
            #     t.write('zd_q:'+str(zd_q)+'\n'+'zd_p:'+
            #     str(zd_p)+'\n'+'zx_q:'+str(zx_q)+'\n'+'zx_p'+str(zx_p)+'\n'
            #     +'zy_q:'+str(zy_q)+'\n'
            #     +'zy_p'+str(zy_p)+'\n')

            return   self.beta_y * KLD_y \
                   + self.aux_loss_multiplier_d * CE_d \
                   + self.aux_loss_multiplier_y * CE_y, \
                   CE_y

            # return -self.beta_d * mmd_zdpzdq \
            #        - self.beta_y * mmd_zypzyq \
            #        - self.beta_x * mmd_zxpzxq \
            #        + self.aux_loss_multiplier_d * CE_d \
            #        + self.aux_loss_multiplier_y * CE_y,\
            #        CE_y


    def classifier(self, x):
        """
        classify an image (or a batch of images)
        :param xs: a batch of scaled vectors of pixels from an image
        :return: a batch of the corresponding class labels (as one-hots)
        """
        with torch.no_grad():
            zd_mu, zd_sigma, logit_d = self.qzd.forward(x)
            zy_mu, zy_sigma = self.zd_to_zy.forward(zd_mu, zd_sigma)
            feature, feature_plus, y_hat = self.py(x, zy_mu)
            alpha_y = F.softmax(y_hat, dim=1)

            # get the index (digit) that corresponds to
            # the maximum predicted class probability
            res, ind = torch.topk(alpha_y, 1)

            # convert the digit(s) to one-hot tensor(s)
            y = x.new_zeros(alpha_y.size())
            y = y.scatter_(1, ind, 1.0)

            alpha_d = F.softmax(logit_d, dim=1)

            # get the index (digit) that corresponds to
            # the maximum predicted class probability
            res, ind = torch.topk(alpha_d, 1)

            # convert the digit(s) to one-hot tensor(s)
            d = x.new_zeros(alpha_d.size())
            d = d.scatter_(1, ind, 1.0)

        return d, y
