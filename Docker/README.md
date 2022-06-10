## Build your Docker for SPATL

This image is built based on [Docker Pytorch](https://github.com/anibali/docker-pytorch), thanks very much for their
great efforts.

### Requirements


In order to use this image you must have Docker Engine installed. Instructions
for setting up Docker Engine are
[available on the Docker website](https://docs.docker.com/engine/installation/).

#### CUDA requirements 

If you have a CUDA-compatible NVIDIA graphics card, you can use a CUDA-enabled
version of the PyTorch image to enable hardware acceleration. 

Firstly, ensure that you install the appropriate NVIDIA drivers. On Ubuntu,
I've found that the easiest way of ensuring that you have the right version
of the drivers set up is by installing a version of CUDA _at least as new as
the image you intend to use_ via
[the official NVIDIA CUDA download page](https://developer.nvidia.com/cuda-downloads).
As an example, if you intend on using the `cuda-10.1` image then setting up
CUDA 10.1 or CUDA 10.2 should ensure that you have the correct graphics drivers.

You will also need to install the NVIDIA Container Toolkit to enable GPU device
access within Docker containers. This can be found at
[NVIDIA/nvidia-docker](https://github.com/NVIDIA/nvidia-docker).


### Build your images for SPATL

Go to where the Dockerfile located `Docker/Dockerfile`. The Docker configuration is written in the Dockerfile.

To build your local image, you could simply run the following command:

```bash
$  docker build --tag spatl .
```
To see a list of images we have on our local machine,  simply run the `docker images` command.



### Usage

#### Running PyTorch scripts

It is possible to run PyTorch programs inside a container using the
`python3` command. Go to the directory where the `spatl_federated_learning.py` located.

To conduct the experiment shown in the paper, you could run it with
the following command:

```sh
docker run --rm -it --init \
  --gpus=all \
  --ipc=host \
  --user="$(id -u):$(id -g)" \
  --volume="$PWD:/app" \
  spatl python3 spatl_federated_learning.py --model=vgg --dataset=cifar10 --alg=gradient_control --lr=0.01 --batch-size=64 --epochs=5 --n_parties=30 --beta=0.1 --device='cuda' --datadir='./data/' --logdir='./logs/'  --noise=0 --sample=0.7 --rho=0.9 --comm_round=200 --init_seed=0
```

Here's a description of the Docker command-line options shown above:

* `--gpus=all`: Required if using CUDA, optional otherwise. Passes the
  graphics cards from the host to the container. You can also more precisely
  control which graphics cards are exposed using this option (see documentation
  at https://github.com/NVIDIA/nvidia-docker).
* `--ipc=host`: Required if using multiprocessing, as explained at
  https://github.com/pytorch/pytorch#docker-image.
* `--user="$(id -u):$(id -g)"`: Sets the user inside the container to match your
  user and group ID. Optional, but is useful for writing files with correct
  ownership.
* `--volume="$PWD:/app"`: Mounts the current working directory into the container.
  The default working directory inside the container is `/app`. Optional.

#### Running different experiment

If you are going to reproduce different experiment results shown in the paper, you can
change the hyper-parameter arguments, you can find all the listed arguments in the `utils/parameter.py`.
To run different experiment, you can change the argument in ```python3 spatl_federated_learning.py --xxx xxx ```


