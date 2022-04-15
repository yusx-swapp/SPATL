# SPATL 
SPATL: Salient Parameter Aggregation and Transfer Learning for Heterogeneous Clients in Federated Learning
## Dependencies

Current code base is tested under following environment:

1. Python   3.8
2. PyTorch  1.8.0 (cuda 11.1)
3. torchvision 0.7.0
4. [torch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#) 1.6.1

## Efficient federated learning
SPATL Overview. SPATL trains a shared encoder through federated learning, and transfers the  knowledge to heterogeneous clients. Clients upload salient parameters selected by a pre-trained RL-agent. The selected parameters are then aggregated by the server.
![](./logs/figure/overview.png)

In this work, SPATL performs efficient federated learning throught salient parameter aggregation, transfer learning, and gradient control. We test SPATL
on ResNet20, ResNet32, VGG-11, and 2-layer simple CNN.
### Non-IID CIFAR-10
In this subsection, clients are trained on CIFAR-10 with Non-IID settings.

Train ResNet-32 200 rounds with 10 clients and sample ratio = 1:
   ```
python spatl_federated_learning.py --model=ResNet-32 --dataset=cifar10 --alg=gradient_control --lr=0.01 --batch-size=64 --epochs=5 --n_parties=30 --beta=0.1 --device='cuda' --datadir='./data/' --logdir='./logs/'  --noise=0 --sample=1 --rho=0.9 --comm_round=200 --init_seed=0
   ```
Train vgg-11 200 rounds with 30 clients and sample ratio = 0.7:
  ```angular2html
python spatl_federated_learning.py --model=vgg --dataset=cifar10 --alg=gradient_control --lr=0.01 --batch-size=64 --epochs=5 --n_parties=30 --beta=0.1 --device='cuda' --datadir='./data/' --logdir='./logs/'  --noise=0 --sample=0.7 --rho=0.9 --comm_round=200 --init_seed=0
   ```
Federated learning results (Compare with SoTAs):
![](./logs/figure/train_effi.png)
Under different experiment settings:
![](./logs/figure/train_effi_2.png)

Communication cost savings to reach the target accuracy:
![](./logs/figure/com_cost.png)
