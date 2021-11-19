import json
import logging
import os
import time
import torch

import datetime

import numpy as np
from torch.utils import data

from federated_learning.network_training import train_net, local_update, local_update_scaffold
from pruning_head.gnnrl_network_pruning import get_num_hidden_layer
from pruning_head.graph_env.graph_environment import graph_env
from utils.load_neural_networks import init_nets
from utils.log_utils import mkdirs
from utils.parameters import get_parameter
from utils.data.prepare_data import partition_data, get_dataloader
from utils.save_model import save_checkpoint







#
# def get_partition_dict(dataset, partition, n_parties, init_seed=0, datadir='./data', logdir='./logs', beta=0.5):
#     seed = init_seed
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
#         dataset, datadir, logdir, partition, n_parties, beta=beta)
#
#     return net_dataidx_map

if __name__ == '__main__':

    args = get_parameter()


    ###################################################################################################

    mkdirs(args.logdir)
    # mkdirs(args.modeldir)
    if args.log_file_name is None:
        argument_path='experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S")
    else:
        argument_path=args.log_file_name+'.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)


    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S"))
    log_path=args.log_file_name+'.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        # filename='/home/qinbin/test.log',
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # logger.setLevel(logging.INFO)
    device = torch.device(args.device)
    logger.info(device)
    ###################################################################################################
    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    logger.info("Partitioning data")

    '''
    prepare data
    '''
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.datadir, args.logdir, args.partition, args.n_parties, beta=args.beta)
    n_classes = len(np.unique(y_train))
    train_dl_global, test_dl_global, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                                      args.datadir,
                                                                                      args.batch_size,
                                                                                      32)
    # print("len train_dl_global:", len(train_ds_global))


    data_size = len(test_ds_global)

    train_all_in_list = []
    test_all_in_list = []

    if args.noise > 0:
        for party_id in range(args.n_parties):
            dataidxs = net_dataidx_map[party_id]

            noise_level = args.noise
            if party_id == args.n_parties - 1:
                noise_level = 0

            if args.noise_type == 'space':
                train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level, party_id, args.n_parties-1)
            else:
                noise_level = args.noise / (args.n_parties - 1) * party_id
                train_dl_local, test_dl_local, train_ds_local, test_ds_local = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs, noise_level)
            train_all_in_list.append(train_ds_local)
            test_all_in_list.append(test_ds_local)
        train_all_in_ds = data.ConcatDataset(train_all_in_list)
        train_dl_global = data.DataLoader(dataset=train_all_in_ds, batch_size=args.batch_size, shuffle=True)
        test_all_in_ds = data.ConcatDataset(test_all_in_list)
        test_dl_global = data.DataLoader(dataset=test_all_in_ds, batch_size=32, shuffle=False)



    if args.alg == 'no_gradient_control':
        logger.info("Initializing nets")
        #todo modify args
        nets, local_model_meta_data, layer_type = init_nets(args.n_parties,args.model, args)
        global_models, global_model_meta_data, global_layer_type = init_nets( 1, args.model,args)
        global_model = global_models[0]

        global_para = global_model.module.encoder.state_dict()

        # set all the edge model with same initial weights
        if args.is_same_initial:
            for net_id, net in nets.items():
                net.module.encoder.load_state_dict(global_para)

        #record time
        t_start = time.time()
        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round)+"#" * 100)

            # select clients
            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

            '''
            global_para = global_model.state_dict()
            if round == 0:
                if args.is_same_initial:
                    for idx in selected:
                        nets[idx].load_state_dict(global_para)
            else:
                for idx in selected:
                    nets[idx].load_state_dict(global_para)
            '''

            for idx in selected:
                nets[idx].module.encoder.load_state_dict(global_para)
                #local updates:
            # if round == 0:
            #     prune = False
            # else:
            #     prune = True
            # prune = True
            prune = False
            local_update(nets, selected, args, net_dataidx_map,logger, test_dl = None, device=device, Prune=prune)

            # update global model
            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):

                net_para = nets[selected[idx]].cpu().module.encoder.state_dict()
                if idx == 0:
                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]
                else:
                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            global_model.module.encoder.load_state_dict(global_para)

            # logger.info('global n_training: %d' % len(train_dl_global))
            # logger.info('global n_test: %d' % len(test_dl_global))
            #
            #
            # train_acc = compute_accuracy(global_model, train_dl_global)
            # test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True)
            #
            #
            # logger.info('>> Global Model Train accuracy: %f' % train_acc)
            # logger.info('>> Global Model Test accuracy: %f' % test_acc)

        #record time
        t_end = time.time()

        logger.info("total communication time: %f" %(t_end- t_start))
        logger.info("avg time per round: %f" %((t_end- t_start)/args.comm_round))

    elif args.alg == 'gradient_control':


        logger.info("Initializing nets")

        nets, local_model_meta_data, layer_type = init_nets(args.n_parties,args.model, args)

        global_models, global_model_meta_data, global_layer_type = init_nets( 1, args.model,args)

        global_model = global_models[0]

        c_nets, _, _ = init_nets(args.n_parties,args.model, args)

        c_globals, _, _ = init_nets( 1, args.model,args)

        c_global = c_globals[0]

        c_global_para = c_global.module.encoder.state_dict()

        for net_id, net in c_nets.items():
            net.module.encoder.load_state_dict(c_global_para)

        global_para = global_model.module.encoder.state_dict()

        if args.is_same_initial:

            for net_id, net in nets.items():
                net.module.encoder.load_state_dict(global_para)

        # if args.dataset == "cifar10":
        #     input_x = torch.randn([1,3,32,32]).to(device)
        #
        # n_layer,layer_share = get_num_hidden_layer(global_model,args)
        # env = graph_env(global_model,n_layer,args.dataset,test_dl_global,args.compression_ratio,args.g_in_size,args.log_dir,input_x,device,args)

        # record time
        t_start = time.time()
        for round in range(args.comm_round):

            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)

            np.random.shuffle(arr)

            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.module.encoder.state_dict()

            if round == 0:

                if args.is_same_initial:

                    for idx in selected:
                        nets[idx].module.encoder.load_state_dict(global_para)

            else:

                for idx in selected:
                    nets[idx].module.encoder.load_state_dict(global_para)

            # if round == 0:
            #     prune = False
            # else:
            #     prune = True
            prune = True
            # local_update_scaffold(nets, selected, global_model, c_nets, c_global, args, net_dataidx_map,logger,
            #                       env,test_dl=test_dl_global, device=device,Prune=prune)
            local_update_scaffold(nets, selected, global_model, c_nets, c_global, args, net_dataidx_map,logger,
                      test_dl=test_dl_global, device=device,Prune=prune)
            # local_train_net(nets, args, net_dataidx_map, local_split=False, device=device)

            # update global model

            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])

            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):

                net_para = nets[selected[idx]].cpu().module.encoder.state_dict()

                if idx == 0:

                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]

                else:

                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]

            global_model.module.encoder.load_state_dict(global_para)

            # logger.info('global n_training: %d' % len(train_dl_global))
            #
            # logger.info('global n_test: %d' % len(test_dl_global))
            #
            # global_model.to('cpu')
            #
            # train_acc = compute_accuracy(global_model, train_dl_global)
            #
            # test_acc, conf_matrix = compute_accuracy(global_model, test_dl_global, get_confusion_matrix=True)
            #
            # logger.info('>> Global Model Train accuracy: %f' % train_acc)
            #
            # logger.info('>> Global Model Test accuracy: %f' % test_acc)

        # record time
        t_end = time.time()

        logger.info("total communication time: %f" % (t_end - t_start))
        logger.info("avg time per round: %f" % ((t_end - t_start) / args.comm_round))

    #add fl alg here
    elif args.alg== '???':
        #record time
        t_start = time.time()

        '''add code here'''
        #record time
        t_end = time.time()

        '''delete raise NotImplementedError'''
        raise NotImplementedError



    elif args.alg == 'local_training':
        logger.info("Initializing nets")
        nets, local_model_meta_data, layer_type = init_nets(args.n_parties, args)
        arr = np.arange(args.n_parties)
        # local_train_net(nets, arra, args, net_dataidx_map, test_dl = test_dl_global, device=device)

    elif args.alg == 'all_in':
        nets, local_model_meta_data, layer_type = init_nets(1, args)
        n_epoch = args.epochs

        trainacc, testacc = train_net(0, nets[0], train_dl_global, test_dl_global, n_epoch, args.lr, args.optimizer, device=device)

        logger.info("All in test acc: %f" % testacc)


    save_checkpoint({
        'state_dict': global_model.module.state_dict() if isinstance(global_model,
                                                                     torch.nn.DataParallel) else global_model.state_dict(),
        # 'acc': test_acc,

    }, checkpoint_dir="federated_learning/save/vgg/static-pruned-25/iid/")


'''
python spatl_federated_learning.py \
--model=resnet32 \
--dataset=cifar10 \
--alg=fedavg \
--lr=0.01 \
--batch-size=64 \
--epochs=20 \
--n_parties=100 \
--partition=noniid-labeldir \
--beta=0.1 \
--device='cuda' \
--datadir='./data/' \
--logdir='./logs/'  \
--noise=0 \
--sample=0.1 \
--init_seed=0 \

# --train-flag 
--comm_round=100 \
--mu=0.01 \
--rho=0.9 \
scontrol show job 305995
'''

