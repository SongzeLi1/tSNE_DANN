import configargparse
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE

import data_loader
import os
import torch
import models
import utils
from utils import str2bool
import numpy as np
import random


def get_parser():
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="Transfer learning config parser",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    # general configuration
    parser.add("--config", is_config_file=True, help="config file path")
    parser.add("--seed", type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)

    # network related
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--use_bottleneck', type=str2bool, default=True)

    # data loading related
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--src_domain', type=str, required=True)
    parser.add_argument('--tgt_domain', type=str, required=True)

    # training related
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--early_stop', type=int, default=0, help="Early stopping")
    parser.add_argument('--epoch_based_training', type=str2bool, default=False,
                        help="Epoch-based training / Iteration-based training")
    parser.add_argument("--n_iter_per_epoch", type=int, default=20, help="Used in Iteration-based training")

    # optimizer related
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    # learning rate scheduler related
    parser.add_argument('--lr_gamma', type=float, default=0.0003)
    parser.add_argument('--lr_decay', type=float, default=0.75)
    parser.add_argument('--lr_scheduler', type=str2bool, default=True)

    # transfer related
    parser.add_argument('--transfer_loss_weight', type=float, default=10)
    parser.add_argument('--transfer_loss', type=str, default='mmd')
    return parser


def load_data(args):
    '''
    src_domain, tgt_domain data to load
    '''
    folder_src = os.path.join(args.data_dir, args.src_domain)
    folder_tgt = os.path.join(args.data_dir, args.tgt_domain)
    source_loader, n_class = data_loader.load_data(
        folder_src, args.batch_size, infinite_data_loader=not args.epoch_based_training, train=True, num_workers=args.num_workers)
    target_train_loader, _ = data_loader.load_data(
        folder_tgt, args.batch_size, infinite_data_loader=not args.epoch_based_training, train=True, num_workers=args.num_workers)
    target_test_loader, _ = data_loader.load_data(
        folder_tgt, args.batch_size, infinite_data_loader=False, train=False, num_workers=args.num_workers)
    return source_loader, target_train_loader, target_test_loader, n_class

def get_model(args):
    model = models.TransferNet(
        args.n_class, transfer_loss=args.transfer_loss, base_net=args.backbone, max_iter=args.max_iter, use_bottleneck=args.use_bottleneck).to(args.device)
    return model


def test(model, target_test_loader, args):
    model.eval()
    correct = 0
    features = []
    labels = []

    len_target_dataset = len(target_test_loader.dataset)
    with torch.no_grad():
        for data, target in target_test_loader:
            data, target = data.to(args.device), target.to(args.device)
            s_output,feature_output = model.predict(data)  # Assuming s_output is the output from the last layer

            # Collect predictions for accuracy
            pred = torch.max(s_output, 1)[1]
            correct += torch.sum(pred == target).item()

            # Collect features and labels for t-SNE
            features.append(feature_output.cpu().numpy())
            labels.append(target.cpu().numpy())

    # print(labels)
    # Calculating the accuracy
    acc = 100. * correct / len_target_dataset

    # t-SNE Visualization
    labels = np.concatenate(labels, axis=0)
    class_num = len(np.unique(labels))
    # print("Class num:", class_num)

    features = np.concatenate(features, axis=0)
    tsne = TSNE(n_components=2, random_state=0)
    tsne_results = tsne.fit_transform(features)
    x_min, x_max = np.min(tsne_results, 0), np.max(tsne_results, 0)
    tsne_results = (tsne_results - x_min) / (x_max - x_min)

    # Using tab20b and tab20c to create a color map with 40 unique colors
    colors_tab20b = plt.cm.tab20b(np.arange(20))
    colors_tab20c = plt.cm.tab20c(np.arange(20))
    full_color_set = np.vstack((colors_tab20b, colors_tab20c))
    cmap = ListedColormap(full_color_set[:class_num])  # Select only the first 31 colors

    plt.figure(figsize=(8, 8))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap=cmap, alpha=0.8)
    plt.title('W --> A')
    plt.xticks([])  # Remove x-axis labels
    plt.yticks([])  # Remove y-axis labels
    plt.grid(True)
    plt.show()

    return acc


def main():
    parser = get_parser()
    args = parser.parse_args()
    setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(args)
    source_loader, target_train_loader, target_test_loader, n_class = load_data(args)
    setattr(args, "n_class", n_class)
    if args.epoch_based_training:
        setattr(args, "max_iter", args.n_epoch * min(len(source_loader), len(target_train_loader)))
    else:
        setattr(args, "max_iter", args.n_epoch * args.n_iter_per_epoch)
    model = get_model(args)
    model.load_state_dict(torch.load('scripts/checkpoint/DANN_webcam_amazon.pth'))

    test_acc = test(model, target_test_loader, args)
    # print(test_acc)

if __name__ == "__main__":
    main()