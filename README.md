# Nathan-s-Framework
Inspired by PyTorch Example: ImageNet training in PyTorch @ https://github.com/pytorch/examples/tree/master/imagenet

This main.py is for PyTorch version 0.3.1

The code for v 0.4.0 will be pushed soon.


## How to run main.py in terminal
In Terminial, use syntax **run main.py < aruguments >**, where **< arguments >** is represent by **parameter names** and **the exactly settings**:
* the number of data loading workers: use **-j** or **--workers** and followed **by the number of workers**,
  * For example, set 8 worers, use syntax **run main.py -j 8**
  * by default, use 4 workers
* the number of total epochs to run:
  * **--epochs**
  * 90 epoches by default
* manual epoch number (useful on restarts)'
  * **-start-epoch**
  * 0 by default
* mini-batch size
  * **-b** or **--batch-size**
  * 256
* initial learning rate
  * **--lr** or **--learning-rate**
  * 0.1
* momentum
  * **--momentum**
  * 0.9
* weight decay
  * **--wd** or **--weight-decay**
  * 1e-4
* print frequency
  * **-p** or **--print-freq**
  * 10
* path to latest checkpoint (default: none)
  * **--resume**
  * default=''
* evaluate model on validation set
  * **-e**, **--evaluate**, dest='evaluate', action='store_true'
* use pre-trained model
  * **--pretrained**, dest='pretrained', action='store_true'
* number of distributed processes
  * **--world-size**
  * 1
* < url > used to set up distributed training
  * --dist-url
  * default='tcp://10.1.75.35', type=str (revise it if needed)
* set GPU id
  * **--gpu-ids**
  * [0]
* path for loading data
  * **--data-path**
  * './data/RBL8.csv' (revise it if needed)
* path for saving models
  * **--save-path**
  * default='./checkpoints' (revise it if needed)

> Assume one wanna set 8 works, 512 mini-batches, 2000 epochs, learining rate to be 0.05, momentum be 0.95, print frequncy be 100, eavaluating model on:   
> **run main.py -j 8 --b 512 -epochs 2000 --lr 0.05 --momentum 0.95 -p 100 -e**


## Explaination in details

All the agruments are corresponding to the code snippet of the parameter setting. 

* parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
   * both **-j** and **--workers** are the name of the parameter
   * once you want to set **the number of data loading workers**, use the argument:
      * python main.py -j 2 (use 2 workers for instance), or
      * python main.py --workers 8 (use 8 workers)
* parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
* parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
* parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
* parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
* parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
* parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
* parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
* parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
* parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
* parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
* parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
* parser.add_argument('--dist-url', default='tcp://10.1.75.35', type=str,
                    help='url used to set up distributed training')
* parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
* parser.add_argument('--gpu-ids', type=int, nargs="+", default=[0],
                    help='gpu ids')
* parser.add_argument('--data-path', type=str, default='./data/RBL8.csv',
                    help='path for data')
* parser.add_argument('--save-path', type=str, default='./checkpoints',
                    help='path for saving models')
