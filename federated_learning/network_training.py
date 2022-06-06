import copy

import torch
from sklearn.metrics import confusion_matrix
from torch import optim, nn
import numpy as np
from pruning_head.gnnrl_network_pruning import gnnrl_pruning
from pruning_head.graph_env.network_pruning import channel_pruning
from utils.accuracy import compute_acc
from utils.data.prepare_data import get_dataloader
from utils.loss import LossCalculator
from torch.nn.utils import prune

################################fedavg################################
def local_update(nets, selected, args, net_dataidx_map,logger, lr=0.01,test_dl = None, device="cpu", Prune=True):
    avg_acc = 0.0
    pre_avg_acc = 0.0
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]

        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
        train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs



        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)




        trainacc, testacc, pre_trainacc, pre_testacc = train_net(net_id, net, train_dl_local, test_dl_local, n_epoch, lr, args.optimizer, logger,args,device=device)
        logger.info("net %d final test acc %f" % (net_id, testacc))

        if Prune:
            logger.info("--------------------------------------Pruning network %s.--------------------------------------" % (str(net_id)))
            net,_,sparsity = gnnrl_pruning(net,logger, test_dl_local,args)
            logger.info("Flops ratio: %s." % (str(_)))
            logger.info("--------------------------------------End pruning %s.--------------------------------------" % (str(net_id)))
            for name, module in net.module.encoder.named_modules(): #remove mask
                if isinstance(module, nn.Conv2d):
                    module = prune.remove(module,name='weight')


        avg_acc += testacc
        pre_avg_acc += pre_testacc
        # saving the trained models here
        # save_model(net, net_id, args)
        # else:
        #     load_model(net, net_id, device=device)

    avg_acc /= len(selected)
    pre_avg_acc /= len(selected)
    # if args.alg == 'local_training':
    #     logger.info("avg test acc %f" % avg_acc)
    logger.info("avg test acc after aggregate %f" % pre_avg_acc)
    logger.info("avg test acc after local update %f" % avg_acc)

    nets_list = list(nets.values())
    return nets_list

    # raise NotImplementedError

def train_net(net_id, net, train_dataloader, test_dataloader, epochs, lr, args_optimizer,logger,args, device="cpu"):
    logger.info('Training network %s' % str(net_id))

    # pre_train_acc = compute_accuracy(net, train_dataloader, device=device)
    # pre_test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    criterion = nn.CrossEntropyLoss().to(device)
    pre_train_acc,_ = compute_acc(train_dataloader, device, net, criterion)
    pre_test_acc, _ = compute_acc(test_dataloader, device, net, criterion)

    logger.info('>> Pre-Training Training accuracy: {}'.format(pre_train_acc/100))
    logger.info('>> Pre-Training Test accuracy: {}'.format(pre_test_acc/100))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
    # criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    #writer = SummaryWriter()
    loss_calculator = LossCalculator()


    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                # loss = criterion(out, target)
                loss = loss_calculator.calc_loss(out, target)

                loss.backward()
                optimizer.step()

                cnt += 1
                epoch_loss_collector.append(loss.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

    # train_acc = compute_accuracy(net, train_dataloader, device=device)
    # test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    train_acc,_ = compute_acc(train_dataloader, device, net, criterion)
    test_acc, _ = compute_acc(test_dataloader, device, net, criterion)
    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)


    logger.info(' ** Training complete **')
    return train_acc, test_acc, pre_train_acc,pre_test_acc


################################Scaffold notransfer################################
def local_update_scaffold_notransfer(nets, selected, global_model, c_nets, c_global, args, net_dataidx_map, logger, test_dl = None, device="cpu", Prune=True):
    avg_acc = 0.0
    pre_avg_acc = 0.0

    total_delta = copy.deepcopy(global_model.module.state_dict())
    for key in total_delta:
        total_delta[key] = 0.0
    c_global.to(device)
    global_model.to(device)
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]


        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
        # train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs


        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        c_nets[net_id].to(device)




        trainacc, testacc,  pre_trainacc, pre_testacc, c_delta_para = train_net_scaffold_notransfer(net_id, net, global_model,c_nets[net_id], c_global, train_dl_local, test_dl_local, n_epoch, args.lr, args.optimizer, logger, args, device=device)

        if Prune:
            # env.reset()
            # env.best_pruned_model=None
            # env.model = net
            logger.info("--------------------------------------Pruning network %s.--------------------------------------" % (str(net_id)))
            # net,_ = gnnrl_pruning(net, env,logger,args)
            net,_,sparsity = gnnrl_pruning(net,logger,test_dl_local,args)
            logger.info("Flops ratio: %s." % (str(_)))
            logger.info("Sparcity of salient parameters: %s." % (str(sparsity)))
            logger.info("--------------------------------------End pruning %s.--------------------------------------" % (str(net_id)))
            for name, module in net.module.named_modules(): #remove mask
                if isinstance(module, nn.Conv2d):
                    module = prune.remove(module,name='weight')

        c_nets[net_id].to('cpu')
        for key in total_delta:
            total_delta[key] += c_delta_para[key]


        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
        pre_avg_acc += pre_testacc

    for key in total_delta:
        total_delta[key] /= len(selected)
    c_global_para = c_global.module.state_dict()
    for key in c_global_para:
        if c_global_para[key].type() == 'torch.LongTensor':
            c_global_para[key] += total_delta[key].type(torch.LongTensor)
        elif c_global_para[key].type() == 'torch.cuda.LongTensor':
            c_global_para[key] += total_delta[key].type(torch.cuda.LongTensor)
        else:
            #print(c_global_para[key].type())
            c_global_para[key] += total_delta[key]
    c_global.module.load_state_dict(c_global_para)

    avg_acc /= len(selected)
    pre_avg_acc /= len(selected)
    # if args.alg == 'local_training':
    #     logger.info("avg test acc %f" % avg_acc)
    logger.info("avg test acc after aggregate %f" % pre_avg_acc)
    logger.info("avg test acc after local update %f" % avg_acc)


    nets_list = list(nets.values())
    return nets_list
def train_net_scaffold_notransfer(net_id, net, global_model, c_local, c_global, train_dataloader, test_dataloader, epochs, lr, args_optimizer,logger, args, device="cpu"):
    logger.info('Training network %s' % str(net_id))

    # train_acc = compute_accuracy(net, train_dataloader, device=device)
    # test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    criterion = nn.CrossEntropyLoss().to(device)
    pre_train_acc,_ = compute_acc(train_dataloader, device, net, criterion)
    pre_test_acc, _ = compute_acc(test_dataloader, device, net, criterion)

    logger.info('>> Pre-Training Training accuracy: {}'.format(pre_train_acc/100))
    logger.info('>> Pre-Training Test accuracy: {}'.format(pre_test_acc/100))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
    # criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    #writer = SummaryWriter()


    c_global_para = c_global.module.state_dict()
    c_local_para = c_local.module.state_dict()

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                loss = criterion(out, target)

                loss.backward()
                optimizer.step()

                net_para = net.module.state_dict()
                for key in net_para:
                    net_para[key] = net_para[key] - args.lr * (c_global_para[key] - c_local_para[key])*0.0005
                net.module.load_state_dict(net_para)

                cnt += 1
                epoch_loss_collector.append(loss.item())


        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))



    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    c_new_para = c_local.module.state_dict()
    c_delta_para = copy.deepcopy(c_local.module.state_dict())
    global_model_para = global_model.module.state_dict()
    net_para = net.module.state_dict()
    for key in net_para:
        c_new_para[key] = c_new_para[key] - c_global_para[key] + (global_model_para[key] - net_para[key]) / (cnt * args.lr)
        c_delta_para[key] = c_new_para[key] - c_local_para[key]
    c_local.module.load_state_dict(c_new_para)


    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)


    logger.info(' ** Training complete **')
    return train_acc, test_acc, pre_train_acc,pre_test_acc, c_delta_para


################################Scaffold################################
def local_update_scaffold(nets, selected, global_model, c_nets, c_global, args, net_dataidx_map, logger, test_dl = None, device="cpu", Prune=True):
    avg_acc = 0.0
    pre_avg_acc = 0.0

    total_delta = copy.deepcopy(global_model.module.encoder.state_dict())
    for key in total_delta:
        total_delta[key] = 0.0
    c_global.to(device)
    global_model.to(device)
    for net_id, net in nets.items():
        if net_id not in selected:
            continue
        dataidxs = net_dataidx_map[net_id]


        noise_level = args.noise
        if net_id == args.n_parties - 1:
            noise_level = 0

        if args.noise_type == 'space':
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, net_id, args.n_parties-1)
        else:
            noise_level = args.noise / (args.n_parties - 1) * net_id
            train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
        # train_dl_global, test_dl_global, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
        n_epoch = args.epochs


        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        c_nets[net_id].to(device)




        trainacc, testacc,  pre_trainacc, pre_testacc, c_delta_para = train_net_scaffold(net_id, net, global_model,c_nets[net_id], c_global, train_dl_local, test_dl_local, n_epoch, args.lr, args.optimizer, logger, args, device=device)

        if Prune:
            # env.reset()
            # env.best_pruned_model=None
            # env.model = net
            logger.info("--------------------------------------Pruning network %s.--------------------------------------" % (str(net_id)))
            # net,_ = gnnrl_pruning(net, env,logger,args)
            net,_,sparsity = gnnrl_pruning(net,logger,test_dl_local,args)
            logger.info("Flops ratio: %s." % (str(_)))
            logger.info("Sparcity of salient parameters: %s." % (str(sparsity)))
            logger.info("--------------------------------------End pruning %s.--------------------------------------" % (str(net_id)))
            for name, module in net.module.named_modules(): #remove mask
                if isinstance(module, nn.Conv2d):
                    module = prune.remove(module,name='weight')

        c_nets[net_id].to('cpu')
        for key in total_delta:
            total_delta[key] += c_delta_para[key]


        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
        pre_avg_acc += pre_testacc

    for key in total_delta:
        total_delta[key] /= len(selected)
    c_global_para = c_global.module.encoder.state_dict()
    for key in c_global_para:
        if c_global_para[key].type() == 'torch.LongTensor':
            c_global_para[key] += total_delta[key].type(torch.LongTensor)
        elif c_global_para[key].type() == 'torch.cuda.LongTensor':
            c_global_para[key] += total_delta[key].type(torch.cuda.LongTensor)
        else:
            #print(c_global_para[key].type())
            c_global_para[key] += total_delta[key]
    c_global.module.encoder.load_state_dict(c_global_para)

    avg_acc /= len(selected)
    pre_avg_acc /= len(selected)
    # if args.alg == 'local_training':
    #     logger.info("avg test acc %f" % avg_acc)
    logger.info("avg test acc after aggregate %f" % pre_avg_acc)
    logger.info("avg test acc after local update %f" % avg_acc)


    nets_list = list(nets.values())
    return nets_list

def train_net_scaffold(net_id, net, global_model, c_local, c_global, train_dataloader, test_dataloader, epochs, lr, args_optimizer,logger, args, device="cpu"):
    logger.info('Training network %s' % str(net_id))

    # train_acc = compute_accuracy(net, train_dataloader, device=device)
    # test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    criterion = nn.CrossEntropyLoss().to(device)
    pre_train_acc,_ = compute_acc(train_dataloader, device, net, criterion)
    pre_test_acc, _ = compute_acc(test_dataloader, device, net, criterion)

    logger.info('>> Pre-Training Training accuracy: {}'.format(pre_train_acc/100))
    logger.info('>> Pre-Training Test accuracy: {}'.format(pre_test_acc/100))

    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=args.rho, weight_decay=args.reg)
    # criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    if type(train_dataloader) == type([1]):
        pass
    else:
        train_dataloader = [train_dataloader]

    #writer = SummaryWriter()


    c_global_para = c_global.module.encoder.state_dict()
    c_local_para = c_local.module.encoder.state_dict()

    for epoch in range(epochs):
        epoch_loss_collector = []
        for tmp in train_dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device)

                optimizer.zero_grad()
                x.requires_grad = True
                target.requires_grad = False
                target = target.long()

                out = net(x)
                loss = criterion(out, target)

                loss.backward()
                optimizer.step()

                net_para = net.module.encoder.state_dict()
                for key in net_para:
                    net_para[key] = net_para[key] - args.lr * (c_global_para[key] - c_local_para[key])/100
                net.module.encoder.load_state_dict(net_para)

                cnt += 1
                epoch_loss_collector.append(loss.item())


        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))



    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    c_new_para = c_local.module.encoder.state_dict()
    c_delta_para = copy.deepcopy(c_local.module.encoder.state_dict())
    global_model_para = global_model.module.encoder.state_dict()
    net_para = net.module.encoder.state_dict()
    for key in net_para:
        c_new_para[key] = c_new_para[key] - c_global_para[key] + (global_model_para[key] - net_para[key]) / (cnt * args.lr)
        c_delta_para[key] = c_new_para[key] - c_local_para[key]
    c_local.module.encoder.load_state_dict(c_new_para)


    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)


    logger.info(' ** Training complete **')
    return train_acc, test_acc, pre_train_acc,pre_test_acc, c_delta_para



def compute_accuracy(model, dataloader, get_confusion_matrix=False, device="cpu"):
    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    if type(dataloader) == type([1]):
        pass
    else:
        dataloader = [dataloader]

    correct, total = 0, 0
    with torch.no_grad():
        for tmp in dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device, dtype=torch.int64)
                out = model(x)
                _, pred_label = torch.max(out.data, 1)

                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()

                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                else:
                    pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                    true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    if was_training:
        model.train()

    if get_confusion_matrix:
        return correct / float(total), conf_matrix

    return correct / float(total)

