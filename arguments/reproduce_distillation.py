def set_arguments(args):
    if args.dataset == 'imagenet':
        args.data_name = "{}{}".format(args.dataset, args.nclass)
    else:
        args.data_name = "{}".format(args.dataset)
    args.pretrain_dir = './pretrain_models_{}'.format(args.dataset)


    if args.dataset != 'imagenet':
        args.net_type = 'convnet'
        args.depth = 3
        args.niter = 10000
        args.lr_img = 1.0 * args.ipc
        args.pretrain_amount = 20
        args.pretrain_epochs = 60
        if args.dataset[:5] == 'cifar':
            args.data_dir = './data'
        elif args.dataset == 'svhn':
            args.data_dir = ''

        elif args.dataset == 'fashion':
            args.data_dir = ''
        elif args.dataset == 'mnist':
            args.data_dir = ''
        elif args.dataset == 'tinyimagenet':
            args.data_dir = ''
            args.depth = 4
        elif args.dataset in ['imagenette', 'imagewoof', 'imagemeow', 'imagesquawk', 'imagefruit', 'imageyellow']:
            args.factor == 3
            args.lr_img = 10.0 * args.ipc
            args.imagenet_dir = ''
            args.net_type = 'convnet'
            args.depth = 5
            args.niter = 10000
            args.pretrain_amount = 5
            args.pretrain_epochs = 80

            if args.nclass == 10:
                args.batch_real = 128
        else:
            raise AssertionError("Not supported dataset!")
    else:
        args.imagenet_dir = ''
        args.net_type = 'resnet_ap'
        args.depth = 10
        args.niter = 10000


        if args.nclass == 10:
            args.batch_real = 128
            


        if args.factor >= 3 and args.ipc >= 20:
            args.decode_type = 'bound'


    # Result folder name
    if args.test:
        args.save_dir = './test_results/'
    else:
        args.save_dir = './results/{}/{}/ipc{}/lambdac{}_eps{}_seed{}'.format(args.dataset, args.type, args.ipc, str(args.lambdac), str(args.eps), args.seed) 

    return args
