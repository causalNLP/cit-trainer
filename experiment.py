import argparse
import torch
from icecream import ic

def do_IRM_experiment(args, output_path):
    from data import get_simulatedDataset_IRM
    dataset = get_simulatedDataset_IRM(n = args.dataset_size, group_number = args.dataset_group)
    ic(dataset)
    from trainer import Trainer
    trainer = Trainer(dataset, ['x_1', 'x_2', 'env'],
                      [['input', 'x_1'], ['input', 'x_2'], ['input', 'env']],
                      [['x_1', 'x_2', 'y'], ['env', 'x_2', 'y']],
                      lambda_ci=args.lambda_ci, lambda_R=args.lambda_R, output_dir=output_path)
    if (torch.cuda.is_available()):
        trainer.cuda()
    trainer.train_epochs(args.num_epochs, 16, 128)
    loss, R_loss, CI_loss, pvalue, pred, target = trainer.evaluate(16, 128)
    ic(loss, CI_loss, pvalue)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='simulated_IRM')
    #TODO, other experiments
    parser.add_argument('--lambda_ci', type=float, default=0.01)
    parser.add_argument('--lambda_R', type=float, default=1)
    parser.add_argument('--dataset_size', type=int, default=1280)
    parser.add_argument('--dataset_group', type=int, default=1) # using different subgroup in generating dataset
    parser.add_argument('--output_path', type=str, default='./result/')
    parser.add_argument('--num_epochs', type=int, default=5) # using different subgroup in generating dataset
    # default is MSE, other is .... (TODO)
    args = parser.parse_args()
    output_path = args.output_path + args.exp_name + '/' + str(args.dataset_size) + '_' \
                  + str(args.dataset_group) + '_' + str(args.lambda_ci) + '_' + str(args.lambda_R) + '/'
    ic(output_path)
    import os
    if (not os.path.exists(output_path)):
        os.makedirs(output_path)
    if args.exp_name == 'simulated_IRM':
        do_IRM_experiment(args, output_path)
    else:
        raise NotImplementedError


