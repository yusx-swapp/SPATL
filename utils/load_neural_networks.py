import os

import torch
from torch.nn import Sequential
# from torchvision.models import vgg16, vgg11

from networks import resnet
from networks.resnet import resnet32, resnet20
from networks.simple_cnn import SimpleCNN, SimpleCNNMNIST
from networks.vgg import vgg11, vgg16


def load_model(model_name,data_root):

    if model_name == "resnet56":
        net = resnet.__dict__['resnet56']()
        net = torch.nn.DataParallel(net,list(range(args.n_gpu)))
        path = os.path.join(data_root, "pretrained_models",'resnet56-4bfd9763.th')
        checkpoint = torch.load(path,map_location=device)
        net.load_state_dict(checkpoint['state_dict'])

    elif model_name == "resnet44":
        net = resnet.__dict__['resnet44']()
        net = torch.nn.DataParallel(net,list(range(args.n_gpu)))
        path = os.path.join(data_root, "pretrained_models",'resnet44-014dd654.th')
        checkpoint = torch.load(path,map_location=device)
        net.load_state_dict(checkpoint['state_dict'])

    elif model_name == "resnet110":
        net = resnet.__dict__['resnet110']()
        net = torch.nn.DataParallel(net,list(range(args.n_gpu)))
        path = os.path.join(data_root, "pretrained_models", 'resnet110-1d1ed7c2.th')
        checkpoint = torch.load(path, map_location=device)
        net.load_state_dict(checkpoint['state_dict'])

    elif model_name == "resnet32":
        net = resnet.__dict__['resnet32']()
        net = torch.nn.DataParallel(net,list(range(args.n_gpu)))
        path = os.path.join(data_root, "pretrained_models", 'resnet32-d509ac18.th')
        checkpoint = torch.load(path, map_location=device)
        net.load_state_dict(checkpoint['state_dict'])

    elif model_name == "resnet20":
        net = resnet.__dict__['resnet20']()
        net = torch.nn.DataParallel(net,list(range(args.n_gpu)))
        path = os.path.join(data_root, "pretrained_models", 'resnet20-12fca82f.th')
        checkpoint = torch.load(path, map_location=device)
        net.load_state_dict(checkpoint['state_dict'])

    elif model_name == "vgg16":
        net = models.vgg16(pretrained=True).eval()
        net = torch.nn.DataParallel(net)
    elif model_name == "mobilenetv2":
        from data.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
        print('=> Resuming from checkpoint..')
        path = os.path.join(data_root, "pretrained_models", 'mobilenetv2.pth.tar')
        print(path)
        checkpoint = torch.load(path)
        sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        net.load_state_dict(sd)

        # from data.mobilenetv2 import mobilenetv2
        # net = mobilenetv2()
        #
        # if args.ckpt_path is not None:  # assigned checkpoint path to resume from
        #     print('=> Resuming from checkpoint..')
        #     path = os.path.join(args.ckpt_path, args.model+'ckpt.best.pth.tar')
        #     #path = args.ckpt_path
        #     checkpoint = torch.load(path)
        #     sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        #     net.load_state_dict(sd)

        net = torch.nn.DataParallel(net)

    elif model_name == "mobilenet":
        # from data.mobilenet import MobileNet
        # net = MobileNet(n_class=1000)
        # sd = torch.load("data/checkpoints/mobilenet_imagenet.pth.tar")
        # if 'state_dict' in sd:  # a checkpoint but not a state_dict
        #     sd = sd['state_dict']
        # sd = {k.replace('module.', ''): v for k, v in sd.items()}
        # net.load_state_dict(sd)
        # net = net.cuda()
        # net = torch.nn.DataParallel(net)
        from data.mobilenet import mobilenet
        net = mobilenet()
        print('=> Resuming from checkpoint..')
        path = os.path.join(data_root, "pretrained_models", 'mobilenet.pth.tar')
        checkpoint = torch.load(path)
        sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        net.load_state_dict(sd)
        net = torch.nn.DataParallel(net)
    elif model_name == 'shufflenet':
        from data.shufflenet import shufflenet
        net = shufflenet()
        print('=> Resuming from checkpoint..')
        path = os.path.join(data_root, "pretrained_models", 'shufflenetbest.pth.tar')
        checkpoint = torch.load(path)
        sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        net.load_state_dict(sd)
        net = torch.nn.DataParallel(net)
    elif model_name == 'shufflenetv2':
        from data.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
        print('=> Resuming from checkpoint..')
        path = os.path.join(data_root, "pretrained_models", 'shufflenetv2.pth.tar')
        checkpoint = torch.load(path)
        sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        net.load_state_dict(sd)
        net = torch.nn.DataParallel(net)

    else:
        raise KeyError
    return net

def init_nets(n_parties,model_name, args):

    nets = {net_i: None for net_i in range(n_parties)}

    for net_i in range(n_parties):

        if model_name == "vgg":
            net = vgg11()

        # if model_name == "resnet56":
        #     net = resnet.__dict__['resnet56']()


        elif model_name == "resnet44":
            net = resnet.__dict__['resnet44']()


        elif model_name == "resnet110":
            net = resnet.__dict__['resnet110']()


        elif model_name == "resnet32":
            net = resnet32()


        elif model_name == "resnet20":
            net = resnet20()


        elif args.model == "vgg16":
            net = vgg16()

        elif args.model == "simple-cnn":
            if args.dataset in ("cifar10", "cinic10", "svhn"):
                net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=10)
            elif args.dataset in ("mnist", 'femnist', 'fmnist'):
                net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
            elif args.dataset == 'celeba':
                net = SimpleCNN(input_dim=(16 * 5 * 5), hidden_dims=[120, 84], output_dim=2)

        elif args.model =='????':
            raise NotImplementedError

        else:
            raise NotImplementedError

        if args.ckpt_path is not None:
            # path = os.path.join(data_root, "pretrained_models",'resnet56-4bfd9763.th')
            checkpoint = torch.load(args.ckpt_path,map_location=args.device)
            sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            net.load_state_dict(sd)

        net = torch.nn.DataParallel(net)
        nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type


if __name__ == '__main__':
    net = resnet.__dict__['resnet20']()
    net = torch.nn.DataParallel(net)
    print(net.module.encoder)