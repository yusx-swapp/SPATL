import json
import logging
import os
import time
import torch

import datetime

import numpy as np
from torch.utils import data

from federated_learning.network_training import local_update, local_update_scaffold, \
    local_update_scaffold_notransfer
from utils.load_neural_networks import init_nets
from utils.log_utils import mkdirs
from utils.parameters import get_parameter
from utils.data.prepare_data import partition_data, get_dataloader
from utils.save_model import save_checkpoint


#Todo: Learning rate should decay

if __name__ == '__main__':

    args = get_parameter()

    mkdirs(args.logdir)
    if args.log_file_name is None:
        argument_path=args.model+'_'+str(args.n_parties) +'_sample:'+str(args.sample)+'_'+args.alg+'arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S")
    else:
        argument_path=args.log_file_name+'.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)


    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if args.log_file_name is None:
        args.log_file_name = args.model+'_'+str(args.n_parties) +'_sample:'+str(args.sample)+'_'+args.alg+'_experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S"))
    log_path=args.log_file_name+'.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    device = torch.device(args.device)
    logger.info(device)
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



    if args.alg == 'spatl':
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
        lr = args.lr
        for round in range(args.comm_round):
            logger.info("in comm round:" + str(round)+"#" * 100)

            # select clients
            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]


            for idx in selected:
                nets[idx].module.encoder.load_state_dict(global_para)
            prune = True
            local_update(nets, selected, args, net_dataidx_map,logger, lr,test_dl = None, device=device, Prune=prune)
            lr = lr * 0.99
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


        #record time
        t_end = time.time()

        logger.info("total communication time: %f" %(t_end- t_start))
        logger.info("avg time per round: %f" %((t_end- t_start)/args.comm_round))

    elif args.alg == 'scaffold':


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


        # record time
        t_end = time.time()
        torch.save(global_model.module.state_dict(), args.ckp_dir+args.model+str(args.n_parties) +'_sample:'+str(args.sample)+'_'+args.alg+'.pth')
        logger.info("total communication time: %f" % (t_end - t_start))
        logger.info("avg time per round: %f" % ((t_end - t_start) / args.comm_round))

    elif args.alg == 'noselect':


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


            prune = False
            local_update_scaffold(nets, selected, global_model, c_nets, c_global, args, net_dataidx_map,logger,
                                  test_dl=test_dl_global, device=device,Prune=prune)


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


        # record time
        t_end = time.time()
        torch.save(global_model.module.state_dict(), args.ckp_dir+args.model+str(args.n_parties) +'_sample:'+str(args.sample)+'_'+args.alg+'.pth')
        logger.info("total communication time: %f" % (t_end - t_start))
        logger.info("avg time per round: %f" % ((t_end - t_start) / args.comm_round))


    elif args.alg == 'notransfer':


        logger.info("Initializing nets")

        nets, local_model_meta_data, layer_type = init_nets(args.n_parties,args.model, args)

        global_models, global_model_meta_data, global_layer_type = init_nets( 1, args.model,args)

        global_model = global_models[0]

        c_nets, _, _ = init_nets(args.n_parties,args.model, args)

        c_globals, _, _ = init_nets( 1, args.model,args)

        c_global = c_globals[0]

        c_global_para = c_global.module.state_dict()

        for net_id, net in c_nets.items():
            net.module.load_state_dict(c_global_para)

        global_para = global_model.module.state_dict()

        if args.is_same_initial:

            for net_id, net in nets.items():
                net.module.load_state_dict(global_para)


        # record time
        t_start = time.time()
        for round in range(args.comm_round):

            logger.info("in comm round:" + str(round))

            arr = np.arange(args.n_parties)

            np.random.shuffle(arr)

            selected = arr[:int(args.n_parties * args.sample)]

            global_para = global_model.module.state_dict()
            print("load param\n\n\n")
            global_model.module.load_state_dict(global_para)
            if round == 0:

                if args.is_same_initial:

                    for idx in selected:
                        nets[idx].module.load_state_dict(global_para)

            else:

                for idx in selected:
                    nets[idx].module.load_state_dict(global_para)


            prune = True
            local_update_scaffold_notransfer(nets, selected, global_model, c_nets, c_global, args, net_dataidx_map,logger,
                                  test_dl=test_dl_global, device=device,Prune=prune)
            # update global model

            total_data_points = sum([len(net_dataidx_map[r]) for r in selected])

            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in selected]

            for idx in range(len(selected)):

                net_para = nets[selected[idx]].cpu().module.state_dict()

                if idx == 0:

                    for key in net_para:
                        global_para[key] = net_para[key] * fed_avg_freqs[idx]

                else:

                    for key in net_para:
                        global_para[key] += net_para[key] * fed_avg_freqs[idx]
            print("load param 22222\n\n\n")
            global_model.module.load_state_dict(global_para)


        # record time
        t_end = time.time()
        torch.save(global_model.module.state_dict(), args.ckp_dir+args.model+str(args.n_parties) +'_sample:'+str(args.sample)+'_'+args.alg+'.pth')
        logger.info("total communication time: %f" % (t_end - t_start))
        logger.info("avg time per round: %f" % ((t_end - t_start) / args.comm_round))

    #add fl alg here

    #





'''
python multi_head_federated_learning.py \
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

