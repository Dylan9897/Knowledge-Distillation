
import sys
sys.path.append('/mnt/d/资源/Github/知识蒸馏')
from train.train_TextCNN import *
from utils.dataloader import build_dataset,build_iterator
import argparse


if __name__ == '__main__':
    import os

    parser = argparse.ArgumentParser(description="Chinese Text Classification")
    parser.add_argument("--activate", type=str, default=False, help="choose an activate function,default is Mish")
    parser.add_argument('--embedding', default="random", type=str, help="random or pretrain")
    parser.add_argument('--optim', default=False, help="choose an optim function:[SGD,Adagrad,RMSProp,Adadelta,Adam]")
    parser.add_argument('--init', default=False, type=str, help="choose a method for init model:[xavier,kaiming]")
    args = parser.parse_args()
    from utils.utils import *
    from utils.log import logger
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.backends.cudnn.deterministic = True  #

    embedding = args.embedding
    x = import_module('module.' + 'TextCNN')
    config = x.Config(DATA_DIR,embedding)
    start_time = time.time()
    logger.info("Loading data...")
    file_dir = os.path.join(DATA_DIR, 'labeled_data.csv')
    vocab, train_data, dev_data, test_data = build_dataset()
    logger.info("Finish Loading data...")
    logger.info("Build train iters...")
    train_iter = build_iterator(train_data,vocab)
    dev_iter = build_iterator(dev_data,vocab)
    test_iter = build_iterator(test_data,vocab)
    logger.info("Train iters is Done ...")
    config.n_vocab = len(vocab)
    print(config.n_vocab)
    config.init_method = args.init
    config.activate = args.activate
    config.optimizer = args.optim
    model = x.Model(config).to(config.device)
    logger.info("train in {} device".format(config.device))
    logger.info('start initial network parameters')
    init_network(model,config)
    logger.info('finish initial network parameters')

    train(config, model, train_iter, dev_iter, test_iter)


