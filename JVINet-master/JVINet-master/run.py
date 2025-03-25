from audioop import mul
from re import M
import sys
sys.path.insert(0, "./../../../")

import argparse

import numpy as np

import torch
from torch.nn import functional as F
import torch.optim as optim
from torchvision.utils import save_image
import torch.utils.data as data_utils

from model_JVINet import JVINet
from dataset.mnist_loader import MnistRotated

from sklearn import manifold
import matplotlib.pyplot as plt


def train(train_loader, model, optimizer, epoch):
    model.train()
    train_loss = 0
    epoch_class_y_loss = 0

    for batch_idx, (x, y, d) in enumerate(train_loader):
        # To device
        x = x.to(device)
        y = y.to(device)
        d = d.to(device)


        optimizer.zero_grad()
        loss, class_y_loss = model.loss_function(d, x, y)
        loss.backward()
        optimizer.step()

        train_loss += loss
        epoch_class_y_loss += class_y_loss

    train_loss /= len(train_loader.dataset)
    epoch_class_y_loss /= len(train_loader.dataset)

    return train_loss, epoch_class_y_loss



def get_accuracy(data_loader, classifier_fn, batch_size, train=True):
    model.eval()
    """
    compute the accuracy over the supervised training set or the testing set
    """
    predictions_d, actuals_d, predictions_y, actuals_y = [], [], [], []

    with torch.no_grad():
        # use the right data loader
        if train:

            for (xs, ys, ds) in data_loader:

                # To device
                xs, ys, ds = xs.to(device), ys.to(device), ds.to(device)

                # use classification function to compute all predictions for each batch
                pred_d, pred_y = classifier_fn(xs)
                predictions_d.append(pred_d)
                actuals_d.append(ds)
                predictions_y.append(pred_y)
                actuals_y.append(ys)

            # compute the number of accurate predictions
            accurate_preds_d = 0
            for pred, act in zip(predictions_d, actuals_d):
                for i in range(pred.size(0)):
                    v = torch.sum(pred[i] == act[i])
                    accurate_preds_d += (v.item() == 5)

            # calculate the accuracy between 0 and 1
            accuracy_d = (accurate_preds_d * 1.0) / (len(predictions_d) * batch_size)

            # compute the number of accurate predictions
            accurate_preds_y = 0
            for pred, act in zip(predictions_y, actuals_y):
                for i in range(pred.size(0)):
                    v = torch.sum(pred[i] == act[i])
                    accurate_preds_y += (v.item() == 10)

            # calculate the accuracy between 0 and 1
            accuracy_y = (accurate_preds_y * 1.0) / (len(predictions_y) * batch_size)


            return accuracy_d, accuracy_y
        else:

            for (xs, ys, ds) in data_loader:


                # To device
                xs, ys, ds = xs.to(device), ys.to(device), ds.to(device)

                # use classification function to compute all predictions for each batch
                pred_d, pred_y = classifier_fn(xs)
                predictions_d.append(pred_d)
                actuals_d.append(ds)
                predictions_y.append(pred_y)
                actuals_y.append(ys)

            # compute the number of accurate predictions
            accurate_preds_y = 0
            for pred, act in zip(predictions_y, actuals_y):
                for i in range(pred.size(0)):
                    v = torch.sum(pred[i] == act[i])
                    accurate_preds_y += (v.item() == 10)

            # calculate the accuracy between 0 and 1
            accuracy_y = (accurate_preds_y * 1.0) / (len(predictions_y) * batch_size)


            return accuracy_y



if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='TwoTaskVae')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 1)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--num-supervised', default=1000, type=int,
                        help="number of supervised examples, /10 = samples per class")

    # Choose domains
    parser.add_argument('--list_train_domains', type=list, default=['0', '30',  '45', '60', '75'],
                        help='domains used during training')
    parser.add_argument('--list_test_domain', type=str, default='15',
                        help='domain used during testing')

    # Model
    parser.add_argument('--d_dim', type=int, default=5,
                        help='number of classes')
    parser.add_argument('--x_dim', type=int, default=784,
                        help='input size after flattening')
    parser.add_argument('--y_dim', type=int, default=10,
                        help='number of classes')
    parser.add_argument('--zd_dim', type=int, default=64,
                        help='size of latent space 1')
    parser.add_argument('--zx_dim', type=int, default=64,
                        help='size of latent space 2')
    parser.add_argument('--zy_dim', type=int, default=64,
                        help='size of latent space 3')

    # Aux multipliers
    parser.add_argument('--aux_loss_multiplier_y', type=float, default=1.,
                        help='multiplier for y classifier')
    parser.add_argument('--aux_loss_multiplier_d', type=float, default=1.,
                        help='multiplier for d classifier')
    # Beta VAE part
    parser.add_argument('--beta_d', type=float, default=1.,
                        help='multiplier for KL d')
    parser.add_argument('--beta_x', type=float, default=1.,
                        help='multiplier for KL x')
    parser.add_argument('--beta_y', type=float, default=1.,
                        help='multiplier for KL y')
    parser.add_argument('--beta_n', type=float, default=1.,
                        help='multiplier for KL y')

    parser.add_argument('-w', '--warmup', type=int, default=100, metavar='N',
                        help='number of epochs for warm-up. Set to 0 to turn warmup off.')
    parser.add_argument('--max_beta', type=float, default=1., metavar='MB',
                        help='max beta for warm-up')
    parser.add_argument('--min_beta', type=float, default=0.0, metavar='MB',
                        help='min beta for warm-up')

    parser.add_argument('--outpath', type=str, default='./',
                        help='where to save')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': False} if args.cuda else {}

    # Model name
    args.list_test_domain = [args.list_test_domain]
    print(args.outpath)


    # Choose training domains
    all_training_domains = ['0', '15', '30', '45', '60', '75']
    all_training_domains.remove(args.list_test_domain[0])
    args.list_train_domains = all_training_domains

    print(args.list_test_domain, args.list_train_domains)

    # Set seed
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)

    # Load supervised training
    train_loader = data_utils.DataLoader(
        MnistRotated(args.list_train_domains, args.list_test_domain, args.num_supervised, args.seed, './dataset/',
                     train=True),
        batch_size=args.batch_size,
        shuffle=True, **kwargs)

    test_loader = data_utils.DataLoader(
        MnistRotated(args.list_train_domains, args.list_test_domain, args.num_supervised, args.seed, './dataset/',
                     train=False),
        batch_size=args.batch_size,
        shuffle=True, **kwargs)

    # setup the VAE
    model = JVINet(args).to(device)

    # setup the optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_loss = 1000.
    best_y_acc = 0.

    early_stopping_counter = 1
    max_early_stopping = 100

    # training loop
    print('\nStart training:', args)
    for epoch in range(1, args.epochs + 1):
        model.beta_d = min([args.beta_d, args.beta_d * (epoch * 1.) / args.warmup])
        # model.beta_y = min([args.beta_y, args.beta_y * (epoch * 1.) / args.warmup])
        model.beta_x = min([args.beta_x, args.beta_x * (epoch * 1.) / args.warmup])

        # train
        avg_epoch_losses_sup, avg_epoch_class_y_loss = train(train_loader, model, optimizer, epoch)

        # store the loss and validation/testing accuracies in the logfile
        str_loss_sup = avg_epoch_losses_sup
        str_print = "{} epoch: avg loss {}".format(epoch, str_loss_sup)
        str_print += ", class y loss {}".format(avg_epoch_class_y_loss)

        # this test accuracy is only for logging, this is not used
        # to make any decisions during training
        train_accuracy_d, train_accuracy_y = get_accuracy(train_loader, model.classifier, args.batch_size, train=True)
        test_accuracy_y = get_accuracy(test_loader, model.classifier, args.batch_size, train=False)
        str_print += " train accuracy d {}".format(train_accuracy_d)
        str_print += "\n"
        str_print += ", y {}".format(train_accuracy_y)
        str_print += "\n"
        str_print += " test accuracy y {}".format(test_accuracy_y)


        print(str_print)






