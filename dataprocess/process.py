import os
import subprocess

import pandas as pd

from utils.parser import get_parser

parser = get_parser()
option = parser.parse_args()

onweb_path = 'https://data.deepai.org/quora_question_pairs.zip'

ziped_path = os.path.join(option.sources_path, 'quora_question_pairs.zip')
unzip_path = os.path.join(option.sources_path, 'quora_question_pairs')

source_train_path = os.path.join(unzip_path, 'train.csv.zip')
source_train_file = os.path.join(unzip_path, 'train.csv')

subprocess.run('mkdir -p %s' % option.sources_path, shell = True)

if  not os.path.exists(ziped_path):
    os.system( 'wget %s -O %s' % (onweb_path, ziped_path))

if  not os.path.exists(unzip_path):
    os.system('unzip %s -d %s' % (ziped_path, unzip_path))

if  not os.path.exists(source_train_file):
    os.system('unzip %s -d %s' % (source_train_path, unzip_path))

target_train_file = os.path.join(option.targets_path, option.train_file)
target_valid_file = os.path.join(option.targets_path, option.valid_file)
target_test_file  = os.path.join(option.targets_path, option.test_file )

subprocess.run('mkdir -p %s' % option.targets_path, shell = True)

datas = pd.read_csv(source_train_file, sep = ',')

datas = datas[datas['is_duplicate'] == 1]

datas = datas.drop(['is_duplicate', 'qid1', 'qid2'], axis = 1)

datas = datas.rename(columns = {'id': 'id', 'question1': 'q1', 'question2': 'q2'})

train_radio = 0.90
valid_radio = 0.05

train_data_len = int(len(datas) * train_radio)
valid_data_len = int(len(datas) * valid_radio)

train_end_idx = train_data_len
valid_end_idx = train_data_len + valid_data_len

datas[:train_end_idx] \
    .to_csv(target_train_file, sep = '\t', index = None)

datas[train_end_idx:valid_end_idx] \
    .to_csv(target_valid_file, sep = '\t', index = None)

datas[valid_end_idx:] \
    .to_csv(target_test_file , sep = '\t', index = None)
