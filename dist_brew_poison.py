"""General interface script to launch distributed poisoning jobs. Launch only through the pytorch launch utility."""

import socket
import datetime
import time


import torch
import forest
torch.backends.cudnn.benchmark = forest.consts.BENCHMARK
torch.multiprocessing.set_sharing_strategy(forest.consts.SHARING_STRATEGY)

# Parse input arguments
args = forest.options().parse_args()
# Parse training strategy
defs = forest.training_strategy(args)
# 100% reproducibility?
if args.deterministic:
    forest.utils.set_deterministic()

if args.local_rank is None:
    raise ValueError('This script should only be launched via the pytorch launch utility!')


if __name__ == "__main__":

    if torch.cuda.device_count() < args.local_rank:
        raise ValueError('Process invalid, oversubscribing to GPUs is not possible in this mode.')
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device(f'cuda:{args.local_rank}')
    setup = dict(device=device, dtype=torch.float, non_blocking=forest.consts.NON_BLOCKING)
    torch.distributed.init_process_group(backend=forest.consts.DISTRIBUTED_BACKEND, init_method='env://')
    if args.ensemble != 1 and args.ensemble != torch.distributed.get_world_size():
        raise ValueError('Argument given to ensemble does not match number of launched processes!')
    else:
        args.ensemble = torch.distributed.get_world_size()
        if torch.distributed.get_rank() == 0:
            print('Currently evaluating -------------------------------:')
            print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
            print(args)
            print(repr(defs))
            print(f'CPUs: {torch.get_num_threads()}, GPUs: {torch.cuda.device_count()} on {socket.gethostname()}')
            print(f'Ensemble launched on {torch.distributed.get_world_size()} GPUs'
                  f' with backend {forest.consts.DISTRIBUTED_BACKEND}.')

    if torch.cuda.is_available():
        print(f'GPU : {torch.cuda.get_device_name(device=device)}')

    model = forest.Victim(args, setup=setup)
    data = forest.Kettle(args, model.defs.batch_size, model.defs.augmentations, setup=setup)
    witch = forest.Witch(args, setup=setup)

    start_time = time.time()
    if args.pretrained:
        print('Loading pretrained model...')
        stats_clean = None
    else:
        stats_clean = model.train(data, max_epoch=args.max_epoch)
    train_time = time.time()


    poison_delta = witch.brew(model, data)
    brew_time = time.time()

    if not args.pretrained and args.retrain_from_init:
        stats_rerun = model.retrain(data, poison_delta)
    else:
        stats_rerun = None  # we dont know the initial seed for a pretrained model so retraining makes no sense

    if args.vnet is not None:  # Validate the transfer model given by args.vnet
        train_net = args.net
        args.net = args.vnet
        if args.vruns > 0:
            model = forest.Victim(args, setup=setup)
            stats_results = model.validate(data, poison_delta)
        else:
            stats_results = None
        args.net = train_net

    else:  # Validate the main model
        if args.vruns > 0:
            stats_results = model.validate(data, poison_delta)
        else:
            stats_results = None
    test_time = time.time()

    if torch.distributed.get_rank() == 0:
        timestamps = dict(train_time=str(datetime.timedelta(seconds=train_time - start_time)).replace(',', ''),
                          brew_time=str(datetime.timedelta(seconds=brew_time - train_time)).replace(',', ''),
                          test_time=str(datetime.timedelta(seconds=test_time - brew_time)).replace(',', ''))
        # Save run to table
        results = (stats_clean, stats_rerun, stats_results)
        forest.utils.record_results(data, witch.stat_optimal_loss, results,
                                    args, defs, model.model_init_seed, extra_stats=timestamps)

        # Export
        if args.save:
            data.export_poison(poison_delta, path=None, mode='full')

        print(datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p"))
        print('---------------------------------------------------')
        print(f'Finished computations with train time: {str(datetime.timedelta(seconds=train_time - start_time))}')
        print(f'--------------------------- brew time: {str(datetime.timedelta(seconds=brew_time - train_time))}')
        print(f'--------------------------- test time: {str(datetime.timedelta(seconds=test_time - brew_time))}')
        print('-------------Job finished.-------------------------')
