import os
import subprocess

os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

from utils.parser import get_parser
from utils.logger import get_logger

parser = get_parser()
option = parser.parse_args()

root_path = 'result'

logs_folder = os.path.join(root_path, 'logs', option.name)
save_folder = os.path.join(root_path, 'save', option.name)
sample_folder = os.path.join(root_path, 'sample', option.name)
result_folder = os.path.join(root_path, 'result', option.name)

subprocess.run('mkdir -p %s' % logs_folder, shell = True)
subprocess.run('mkdir -p %s' % save_folder, shell = True)
subprocess.run('mkdir -p %s' % sample_folder, shell = True)
subprocess.run('mkdir -p %s' % result_folder, shell = True)

logs_path = os.path.join(logs_folder, 'main.log')
save_path = os.path.join(save_folder, 'best.pth')

logger = get_logger(option.name, logs_path)

from loaders.loader1 import get_loader as get_loader1 # Quora

from modules.module1 import get_module as get_module1 # Seq2Seq

from utils.misc import train, valid, save_checkpoint, load_checkpoint, save_sample

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

logger.info('prepare loader')

src_vocab, trg_vocab, train_loader, valid_loader, test_loader = get_loader1(option)

src_vocab_size = len(src_vocab['word2id'])
trg_vocab_size = len(trg_vocab['word2id'])

src_pad_idx = src_vocab['word2id'].get(src_vocab['special']['PAD_TOKEN'])
trg_pad_idx = trg_vocab['word2id'].get(trg_vocab['special']['PAD_TOKEN'])

logger.info('prepare module')

module = get_module1(option, src_vocab_size, src_pad_idx, trg_vocab_size, trg_pad_idx).to(device)

logger.info('prepare envs')

optimizer = optim.Adam(module.parameters(), lr = option.lr)

criterion = nn.CrossEntropyLoss(ignore_index = trg_pad_idx)

if  option.mode == 'train':
    logger.info('start training!')
    best_valid_loss = float('inf')
    for epoch in range(option.num_epochs):
        train_info = train(module, train_loader, criterion, optimizer, device, option.grad_clip)
        valid_info = valid(module, valid_loader, criterion, optimizer, device, trg_vocab)
        logger.info(
            '[Epoch %d] Train Loss: %.4f, Valid Loss: %.4f, Valid BLEU: %.4f' %
            (epoch, train_info['loss'], valid_info['loss'], valid_info['bleu'])
        )
        if  best_valid_loss > valid_info['loss']:
            best_valid_loss = valid_info['loss']
            save_checkpoint(save_path, module, optimizer, epoch)
            save_sample(sample_folder,
                valid_info['reference_ids'], valid_info['hypothese_ids'],
                valid_info['reference_wds'], valid_info['hypothese_wds'],
            )

if  option.mode == 'test':
    logger.info('start testing!')
    best_epoch = load_checkpoint(save_path, module, optimizer)
    valid_info = valid(module, test_loader, criterion, optimizer, device, trg_vocab)
    logger.info(
        '[Best Epoch %d] Test Loss: %.4f, Test BLEU: %.4f' %
        (best_epoch, valid_info['loss'], valid_info['bleu'])
    )
    save_sample(result_folder,
        valid_info['reference_ids'], valid_info['hypothese_ids'],
        valid_info['reference_wds'], valid_info['hypothese_wds'],
    )
