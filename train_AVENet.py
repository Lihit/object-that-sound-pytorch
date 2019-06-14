import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.optim import *
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import json
import os
from utils import visualize, utils
from dataset import AudioSet
from models import AVENet, AVOLNet
from configs.config_AVENet import config
import pprint
import time
import logging

pprint.pprint(config)
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
vis = visualize.Visualizer(env=config.vis.env, port=config.vis.DEFAULT_PORT, server=config.vis.DEFAULT_HOSTNAME)

time_str = time.strftime("%m-%d-%H-%M", time.localtime())
exp_dir = os.path.join(config.store_dir, config.store_name + '_' + time_str)

if not os.path.exists(exp_dir):
    print("make dirs:%s" % exp_dir)
    os.makedirs(exp_dir)
# add logger config
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(filename=os.path.join(exp_dir, 'train.log'), level=logging.INFO, filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set a format which is simpler for console use
formatter = logging.Formatter('%(levelname)-4s %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)


def validation(model, dataloader, criterion):
    val_loss = utils.AverageMeter()
    val_acc = utils.AverageMeter()
    with torch.no_grad():
        for b_idx, data_dict in enumerate(dataloader):
            image = data_dict['image']
            audio = data_dict['audio']
            target = data_dict['target'].squeeze()
            # Filter the bad ones first
            idx = target != 2
            if idx.sum() == 0:
                continue
            # Find the new variables
            image = Variable(image[idx]).cuda()
            audio = Variable(audio[idx]).cuda()
            target = Variable(target[idx]).cuda()
            out, _, _ = model(image, audio)
            loss = criterion(out, target)
            val_loss.update(loss.item(), image.size(0))
            # Calculate accuracy
            _, ind_max = out.max(1)
            acc = (ind_max.data == target.data).sum().item() / image.size(0)
            val_acc.update(acc, image.size(0))
    return val_acc.avg, val_loss.avg


def train(model, traindataloader, valdataloader, criterion, optimizer, scheduler):
    model.train()
    train_loss = utils.AverageMeter()
    train_acc = utils.AverageMeter()
    for epoch in range(config.train.epochs):
        model.train()
        train_loss.reset()
        train_acc.reset()
        scheduler.step(epoch)
        for b_idx, data_dict in enumerate(traindataloader):
            optimizer.zero_grad()
            image = data_dict['image']
            audio = data_dict['audio']
            target = data_dict['target'].squeeze()
            # Filter the bad ones first
            idx = target != 2
            if idx.sum() == 0:
                continue
            # Find the new variables
            image = Variable(image[idx]).cuda()
            audio = Variable(audio[idx]).cuda()
            target = Variable(target[idx]).cuda()

            out, _, _ = model(image, audio)
            loss = criterion(out, target)
            train_loss.update(loss.item(), image.size(0))
            loss.backward()
            optimizer.step()
            # Calculate accuracy
            _, ind_max = out.max(1)
            acc = (ind_max.data == target.data).sum().item() * 1.0 / image.size(0)
            train_acc.update(acc, image.size(0))
        logging.info("epoch:%d, train accuracy:%0.3f, train loss:%0.3f" % (epoch, train_acc.avg, train_loss.avg))
        if (epoch + 1) % config.train.val_epoch == 0:
            logging.info('start validation>>>')
            val_acc, val_loss = validation(model, valdataloader, criterion)
            logging.info("epoch:%d, validation accuracy:%0.3f, validation loss:%0.3f" % (epoch, val_acc, val_loss))
        if (epoch + 1) % config.train.model_save_epoch == 0:
            logging.info('saving model at %s' % os.path.join(exp_dir, 'model_epoch%d.pth' % epoch))
            torch.save(model.state_dict(), os.path.join(exp_dir, 'model_epoch%d.pth' % epoch))
    logging.info('saving final model at %s' % os.path.join(exp_dir, 'model_final.pth'))
    torch.save(model.state_dict(), os.path.join(exp_dir, 'model_final.pth'))


def crossModalRetrieval(model, testdataloader, mode1="img", mode2="aud", topk_list=[5]):
    with open('./data/data_infos/genre_relevance.csv', 'r') as fin:
        line = fin.readline()
        genre_rel = [i for i in line.replace('\n', '').split(',')]
        mat_rel = np.zeros(shape=(len(genre_rel), len(genre_rel)))
        for i, line in enumerate(fin.readlines()):
            for j, val in enumerate(line.replace('\n', '').split(',')):
                mat_rel[i][j] = int(val)

    assert (mode1 != mode2)
    anchor_feats = []
    positive_feats = []
    anchor_classes = []
    for b_idx, data_dict in enumerate(testdataloader):
        image = data_dict['image']
        audio = data_dict['audio']
        target = data_dict['target'].squeeze()
        vidClasses = data_dict['vidClasses']  # vidClasses is used for the nDCG metrics
        anchor_classes.append(vidClasses)
        # Filter the bad ones first
        idx = target != 2
        if idx.sum() == 0:
            continue
        # Find the new variables
        image = Variable(image[idx]).cuda()
        audio = Variable(audio[idx]).cuda()
        target = Variable(target[idx]).cuda()
        # img_emb = model.get_image_embeddings(image)
        # aud_emb = model.get_audio_embeddings(audio)
        out, img_emb, aud_emb = model(image, audio)
        anchor_feats.append(img_emb.cpu().data.numpy())
        positive_feats.append(aud_emb.cpu().data.numpy())
    anchor_feats = np.concatenate(anchor_feats, 0)
    positive_feats = np.concatenate(positive_feats, 0)

    # topk metrics
    # topk_acc_ret = utils.top_k(anchor_feats, positive_feats, topk_list)
    # for key in topk_acc_ret:
    #     logging.info("%s:%0.3f" % (key, topk_acc_ret[key]))

    # nDCG metrics
    k = 30
    nDCG_score = utils.nDCG_k(anchor_feats, positive_feats, anchor_classes, k, mat_rel, genre_rel)
    logging.info("score of nDCG@%d in test set is: %0.2f" % (k, nDCG_score))


if __name__ == "__main__":
    # define model
    logging.info('start buiding model>>>')
    model = AVENet.AVENet()
    model.cuda()
    logging.info('successfully start buiding model>>>')

    # load save model if exists
    if os.path.exists(config.train.pretrained_model):
        model.load_state_dict(torch.load(config.train.pretrained_model))
        logging.info("Loading from previous checkpoint.")

    # get dataset and dataloader
    traindataset = AudioSet.AudioSetDatasetTrain(config)
    valdataset = AudioSet.AudioSetDatasetVal(config)
    testdataset = AudioSet.AudioSetDatasetTest(config)
    traindataloader = DataLoader(traindataset, batch_size=config.data.batch_size, shuffle=True,
                                 num_workers=config.train.num_workers, drop_last=False)
    valdataloader = DataLoader(valdataset, batch_size=config.data.batch_size, shuffle=False,
                               num_workers=config.train.num_workers, drop_last=False)
    testdataloader = DataLoader(testdataset, batch_size=config.data.batch_size, shuffle=False,
                                num_workers=config.train.num_workers, drop_last=False)

    criterion = torch.nn.CrossEntropyLoss()
    # optim = Adam(model.parameters(), lr=lr, weight_decay=1e-7)
    optimizer = SGD(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train.epochs)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=0.1)

    # train and validation model
    if config.train.is_train:
        train(model, traindataloader, valdataloader, criterion, optimizer, scheduler)

    # validation model
    if config.train.is_val:
        validation(model, valdataloader, criterion)

    # retrieval test
    crossModalRetrieval(model, testdataloader, mode1="img", mode2="aud", topk_list=[5])
