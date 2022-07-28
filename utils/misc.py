import os
import tqdm
import torch

from nltk.translate.bleu_score import corpus_bleu

def save_checkpoint(save_path, model, optim, epoch):
    checkpoint = {
        'model': model.state_dict(),
        'optim': optim.state_dict(),
        'epoch': epoch,
    }
    torch.save(checkpoint, save_path)

def load_checkpoint(load_path, model, optim):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model'])
    optim.load_state_dict(checkpoint['optim'])
    return checkpoint['epoch']

def save_sample(folder, reference_ids, hypothese_ids, reference_wds, hypothese_wds):
    reference_id_path = os.path.join(folder, 'reference_ids.txt')
    hypothese_id_path = os.path.join(folder, 'hypothese_ids.txt')
    reference_wd_path = os.path.join(folder, 'reference_wds.txt')
    hypothese_wd_path = os.path.join(folder, 'hypothese_wds.txt')
    for data, file_path in zip(
        [reference_ids    , hypothese_ids    , reference_wds    , hypothese_wds    ],
        [reference_id_path, hypothese_id_path, reference_wd_path, hypothese_wd_path],
    ):
        with open(file_path, 'w', encoding = 'utf-8') as  text_file:
            text_file.writelines([' '.join([str(item) for item in line]) + '\n' for line in data])

def train(module, loader, criterion, optimizer, device, grad_clip):
    module.train()
    epoch_loss = 0
    for mini_batch in tqdm.tqdm(loader):
        sources , targets, source_length = mini_batch
        sources = sources.to(device)
        targets = targets.to(device)
        outputs = module(sources, source_length, targets, 0.5)
        loss = criterion(
            outputs[:, 1:].reshape(-1, outputs.shape[-1]),
            targets[:, 1:].reshape(-1)
        )
        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), grad_clip)
        optimizer.step()
    return {
        'loss': epoch_loss / len(loader)
    }

def valid(module, loader, criterion, optimizer, device, trg_vocab):
    module.eval()
    epoch_loss = 0
    reference_ids = []
    hypothese_ids = []
    reference_wds = []
    hypothese_wds = []
    SOS_ID = trg_vocab['word2id'].get(trg_vocab['special']['SOS_TOKEN'])
    EOS_ID = trg_vocab['word2id'].get(trg_vocab['special']['EOS_TOKEN'])
    with torch.no_grad():
        for mini_batch in tqdm.tqdm(loader):
            sources , targets, source_length = mini_batch
            sources = sources.to(device)
            targets = targets.to(device)
            outputs = module(sources, source_length, targets, 0)
            loss = criterion(
                outputs[:, 1:].reshape(-1, outputs.shape[-1]),
                targets[:, 1:].reshape(-1)
            )
            epoch_loss += loss.item()
            for word_ids in targets[:, 1:]:
                word_ids = word_ids.tolist()
                word_ids = word_ids[:word_ids.index(EOS_ID) if EOS_ID in word_ids else len(word_ids)]
                reference_ids.append([word_id for word_id in word_ids])
                reference_wds.append([trg_vocab['id2word'].get(word_id) for word_id in word_ids])
            for word_ids in outputs[:, 1:].argmax(-1):
                word_ids = word_ids.tolist()
                word_ids = word_ids[:word_ids.index(EOS_ID) if EOS_ID in word_ids else len(word_ids)]
                hypothese_ids.append([word_id for word_id in word_ids])
                hypothese_wds.append([trg_vocab['id2word'].get(word_id) for word_id in word_ids])
    bleu4 = corpus_bleu([[reference_id] for reference_id in reference_ids], hypothese_ids)
    bleu4 = round(bleu4, 4)
    return {
        'loss': epoch_loss / len(loader),
        'bleu': bleu4,
        'reference_ids': reference_ids,
        'hypothese_ids': hypothese_ids,
        'reference_wds': reference_wds,
        'hypothese_wds': hypothese_wds,
    }
