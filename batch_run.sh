python train.py --ds FashionMNIST --name fmnist_resmcnn --arch resmcnn --gpu=1
python train.py --ds FashionMNIST --name fmnist_mcnn --arch mcnn --gpu=1
python train.py --name mnist_resmcnn --arch resmcnn --gpu=1
python train.py --name mnist_mcnn --arch mcnn --gpu=1
python train.py --ds CIFAR10 --name fmnist_resmcnn --arch resmcnn -c=3 --gpu=1
python train.py --ds CIFAR10 --name fmnist_mcnn --arch mcnn -c=3 --gpu=1