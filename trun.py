
import sys
sys.path.append('/mnt/d/资源/Github/知识蒸馏')
from train.train_TextCNN import *
from utils.dataloader2 import create_data_loader,read_file,build_vocab,build_dataset
import argparse
from transformers import BertTokenizer

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
    x = import_module('module.' + 'TextCNN2')
    config = x.Config(DATA_DIR,embedding)
    start_time = time.time()
    logger.info("Loading data...")


    # config.file_path = "./data/长文本分类/labeled_data.csv"
    # config.RANDOM_SEED = 1
    # config.MAX_VOCAB_SIZE = 10000
    # config.padding_size = 256
    # config.tokenizer = BertTokenizer.from_pretrained("ckpt/bert-base-chinese")


    vocab,df_train,df_val,df_test = build_dataset(config)
    trainloader = create_data_loader(df_train,vocab,config)
    validloader = create_data_loader(df_val,vocab,config)
    testloader = create_data_loader(df_test,vocab,config)

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

    train(config, model, trainloader, validloader, testloader)


