python train.py --ds FashionMNIST --name fmnist_resmcnn28 --arch resmcnn28 --gpu=0
python train.py --ds FashionMNIST --name fmnist_mcnn28 --arch mcnn28 --gpu=0
python train.py --name mnist_resmcnn28 --arch resmcnn28 --gpu=0
python train.py --name mnist_mcnn28 --arch mcnn28 --gpu=0
python train.py --ds CIFAR10 --name cifar10_resmcnn28 --arch resmcnn28 -c=3 --gpu=0
python train.py --ds CIFAR10 --name cifar10_mcnn28 --arch mcnn28 -c=3 --gpu=0
python train.py --ds FashionMNIST --name fmnist_resmcnn16 --arch resmcnn16 --gpu=0
python train.py --ds FashionMNIST --name fmnist_mcnn16 --arch mcnn16 --gpu=0
python train.py --ds FashionMNIST --name fmnist_resmcnn10 --arch resmcnn10 --gpu=0
python train.py --ds FashionMNIST --name fmnist_mcnn10 --arch mcnn10 --gpu=0