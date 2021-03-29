
import argparse

import numpy as np
import torch

from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader

import matplotlib.pyplot as plt

import models
import utils
import time


def arg_parse():
    parser = argparse.ArgumentParser(description='GNN arguments.')
    utils.parse_optimizer(parser)

    parser.add_argument('--model_type', type=str,
                        help='Type of GNN model.')
    parser.add_argument('--batch_size', type=int,
                        help='Training batch size')
    parser.add_argument('--num_layers', type=int,
                        help='Number of graph conv layers')
    parser.add_argument('--hidden_dim', type=int,
                        help='Training hidden size')
    parser.add_argument('--dropout', type=float,
                        help='Dropout rate')
    parser.add_argument('--epochs', type=int,
                        help='Number of training epochs')
    parser.add_argument('--dataset', type=str,
                        help='Dataset')
    parser.add_argument('--examine', action='store_true',
                        help='Examine data')

    parser.set_defaults(model_type='GCN',
                        dataset='cora',
                        num_layers=2,
                        batch_size=32,
                        hidden_dim=32,
                        dropout=0.0,
                        epochs=200,
                        opt='adam',   # opt_parser
                        opt_scheduler='none',
                        weight_decay=0.0,
                        lr=0.01)

    return parser.parse_args()


def train(dataset, task, args):
    global device

    if task == 'graph':
        # graph classification: separate dataloader for test set
        # shuffle dataset before splitting
        data_size = len(dataset)
        idxs = np.arange(data_size).astype(int)
        np.random.shuffle(idxs)
        idxs = list(idxs)
        dataset = dataset[idxs]

        loader = DataLoader(
            dataset[:int(data_size * 0.8)], batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(
            dataset[int(data_size * 0.8):], batch_size=args.batch_size, shuffle=True)
    elif task == 'node':
        # use mask to split train/validation/test
        test_loader = loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True)
    else:
        raise RuntimeError('Unknown task')

    # build model
    model = models.GNNStack(dataset.num_node_features, args.hidden_dim, dataset.num_classes,
                            args, task=task)
    model = model.to(device)
    print(model)
    scheduler, opt = utils.build_optimizer(args, model.parameters())

    # train
    test_accs = []
    best_acc = 0
    timestr = time.strftime("%Y%m%d-%H%M%S")
    for epoch in range(args.epochs):
        total_loss = 0
        model.train()
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            pred = model(batch)
            label = batch.y
            if task == 'node':
                pred = pred[batch.train_mask]
                label = label[batch.train_mask]
            loss = model.loss(pred, label)
            loss.backward()
            opt.step()
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(loader.dataset)
        print(total_loss)

        if epoch % 10 == 0:
            if task == 'graph':
                test_acc = test(test_loader, model)
            else:
                test_acc = test(loader, model, is_validation=True)
            test_accs.append(test_acc)
            print(test_acc, '  test')
            # save best model
            if test_acc > best_acc:
                best_acc = test_acc
            torch.save(model.state_dict(), str(
                args.model_type) + timestr + '.pt')
            # plot accuracies
            x = range(0, epoch+1, 10)
            plt.plot(x, test_accs)
            plt.savefig(str(args.model_type) + timestr + '.png')

    print(f'best achieved accuracy: {best_acc}')
    if model.task == 'node':
        best_model = models.GNNStack(dataset.num_node_features, args.hidden_dim, dataset.num_classes,
                                     args, task=task)
        best_model.load_state_dict(torch.load(str(
            args.model_type) + timestr + '.pt'))
        best_model = best_model.to(device)
        test_acc = test(loader, best_model, is_validation=False)
        print(f'test accuracy: {test_acc}')


def test(loader, model, is_validation=False):
    global device

    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            # max(dim=1) returns values, indices tuple; only need indices
            pred = model(data).max(dim=1)[1]
            label = data.y

        if model.task == 'node':
            mask = data.val_mask if is_validation else data.test_mask
            # node classification: only evaluate on nodes in test set
            pred = pred[mask]
            label = data.y[mask]

        correct += pred.eq(label).sum().item()

    if model.task == 'graph':
        total = len(loader.dataset)
    else:
        total = 0
        for data in loader.dataset:
            total += torch.sum(data.test_mask).item(
            ) if not is_validation else torch.sum(data.val_mask).item()
    return correct / total


def main():
    args = arg_parse()
    np.random.seed(1234)

    if args.dataset == 'enzymes':
        dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
        task = 'graph'
    elif args.dataset == 'cora':
        dataset = Planetoid(root='/tmp/Cora', name='Cora')
        task = 'node'
    if args.examine == True:
        utils.examine_dataset(dataset)

    train(dataset, task, args)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    main()
