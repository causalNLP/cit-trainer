import os
import datasets
def eval_OOD_data():
    pass

def eval_MI():
    pass

lambda_CI_list = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 1]

def load_trainer(output_path, dataset):
    # Always load checkpoint4, which is the last checkpoint
    trainer = Trainer(dataset, ['x_1', 'x_2', 'env'],
                      [['input', 'x_1'], ['input', 'x_2'], ['input', 'env']],
                      [['x_1', 'x_2', 'y'], ['env', 'x_2', 'y']],
                      lambda_ci=0, lambda_R=0, output_dir=output_path)
    trainer.load_checkpoint(output_path + 'epoch_4.pth')
    return trainer

if __name__ == '__main__':
    # eval IRM data
    dataset_path_1 = f"./dataset/simulatedIRMs/128000_1"
    dataset_path_2 = f"./dataset/simulatedIRMs/128000_2"
    dataset_1 = datasets.load_from_disk(dataset_path)
    dataset_2 = datasets.load_from_disk(dataset_path)

    if (os.path.exists(dataset_path)):
        # load huggingface dataset
        dataset = datasets.load_from_disk(dataset_path)
    for lambda_ci in lambda_CI_list:
        output_path = './result/simulated_IRM' + '/' + str(128000) + '_' \
                  + str(1) + '_' + str(lambda_ci) + '_' + str(1) + '/'
        # check if there is result.csv in the output_path
        if not os.path.exists(output_path + 'result.csv'):
            print('No result.csv in ' + output_path)
            continue
        # load the trainer
        trainer = load_trainer(output_path)

    pass