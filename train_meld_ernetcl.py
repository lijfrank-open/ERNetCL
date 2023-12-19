import os
from currlearnloss import CurrLearnLoss
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch.nn.functional as F
import numpy as np, argparse, time, random, math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import MELDRobertaDataset
from loss import AutomaticWeightedLoss, MaskedNLLLoss
from model import ERNet
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

def create_class_weight(mu=1):
    unique = [0, 1, 2, 3, 4, 5, 6]
    labels_dict = {0: 6436, 1: 1636, 2: 358, 3: 1002, 4: 2308, 5: 361, 6: 1607}        
    total = np.sum(list(labels_dict.values()))
    weights = []
    for key in unique:
        score = math.log(mu*total/labels_dict[key])
        weights.append(score)
    return weights

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_MELD_loaders(path='', batch_size=32, classify='emotion', num_workers=0, pin_memory=False):
    trainset = MELDRobertaDataset(path=path, split='train', classify=classify)
    validset = MELDRobertaDataset(path=path, split='valid', classify=classify)
    testset = MELDRobertaDataset(path=path, split='test', classify=classify)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(validset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader

def train_or_eval_model(model, loss_function, loss_function_cl, dataloader, epoch, optimizer=None, train=False, cuda_flag=False,loss_type='',lambd=0.0,epochs=100):
    losses, preds, labels, masks  = [], [], [], []
    vids = []
    max_sequence_len = []

    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()

    seed_everything(seed)
    for iter, data in enumerate(dataloader):
        if train:
            optimizer.zero_grad()

        r1, r2, r3, r4, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda_flag else data[:-1]
        lengths0 = []
        for j, umask_ in enumerate(umask.transpose(0,1)):
            lengths0.append((umask.transpose(0,1)[j] == 1).nonzero()[-1][0] + 1)
        seq_lengths = torch.stack(lengths0)
        
        prob = model(r1, r2, r3, r4, qmask, umask, seq_lengths)

        prob = F.log_softmax(prob,-1)

        prob_ = prob.view(-1, prob.size()[-1])
        label_ = label.view(-1) 

        loss_ = loss_function(prob_, label_, umask)
        
        epochs = 50
        iterations = epoch/epochs
        loss_cl = loss_function_cl(prob, label, umask, qmask, iterations=iterations)

        if loss_type=='auto_loss':
            awl = AutomaticWeightedLoss(2)
            loss = awl(loss_, loss_cl)
        elif loss_type=='sum_loss':
            loss = loss_+ lambd*loss_cl
        elif loss_type=='cl_loss':
            loss = loss_cl
        elif loss_type=='class_loss':
            loss = loss_
        else:
            NotImplementedError

        pred_ = torch.argmax(prob_,1)
        preds.append(pred_.data.cpu().numpy())
        labels.append(label_.data.cpu().numpy())
        masks.append(umask.view(-1).cpu().numpy())
        losses.append(loss.item()*masks[-1].sum())

        if train:
            total_loss = loss
            total_loss.backward()
            if args.tensorboard:
                for param in model.named_parameters():
                    from tensorboardX import SummaryWriter
                    writer = SummaryWriter()
                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()
        else:
            vids += data[-1]

    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks  = np.concatenate(masks)
    else:
        return float('nan'), float('nan'), float('nan'), [], [], [], float('nan'),[]

    avg_loss = round(np.sum(losses)/np.sum(masks), 4)

    avg_accuracy = round(accuracy_score(labels,preds, sample_weight=masks)*100, 2)
    avg_fscore = round(f1_score(labels, preds, sample_weight=masks, average='weighted')*100, 2)
    micro_fscore = round(f1_score(labels, preds, sample_weight=masks, average='micro')*100, 2)
    return avg_loss, avg_accuracy, labels, preds, masks, [avg_fscore, micro_fscore], vids


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', default='/home/lijfrank/code/dataset/MELD_features/meld_features_roberta.pkl', help='dataset dir')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.00001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.0003, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.2, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=128, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=50, metavar='E', help='number of epochs')
    parser.add_argument('--class-weight', action='store_true', default=False, help='use class weights')
    parser.add_argument('--tensorboard', action='store_true', default=False, help='Enables tensorboard log')
    parser.add_argument('--feature_mode', default='concat2', help='concat2')
    parser.add_argument('--seed', type=int, default=2023, metavar='seed', help='seed')
    parser.add_argument('--mu', type=float, default=0, help='class_weight_mu')
    parser.add_argument('--classify', default='emotion',help='sentiment, emotion')
    parser.add_argument('--use_residual', action='store_true', default=True, help='use residual connection')
    parser.add_argument('--rnn_layer', type=int, default=4, help='number of rnn_layer')
    parser.add_argument('--attention_head', type=int, default=4, help='number of attention_head')
    parser.add_argument('--attention_layer', type=int, default=4, help='number of attention_layer')
    parser.add_argument('--input_size', type=int, default=1024, help='')
    parser.add_argument('--input_in_size', type=int, default=1024, help='')
    parser.add_argument('--sigma', type=float, default=0.4, help='cl_loss')
    parser.add_argument('--delt_epoch', type=int, default=10, help='cl_loss')
    parser.add_argument('--loss_type', default='cl_loss', help='auto_loss/sum_loss/cl_loss/class_loss')
    parser.add_argument('--lambd', type=float, default=1.0, help='lambd of loss')
    args = parser.parse_args()
    print(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    if args.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter()

    emo_gru = True
    if args.classify == 'emotion':
        n_classes  = 7
    elif args.classify == 'sentiment':
        n_classes  = 3
    cuda       = args.cuda
    n_epochs   = args.epochs
    batch_size = args.batch_size
    n_speakers = 9

    global seed
    seed = args.seed
    seed_everything(seed)
    
    model = ERNet( rnn_layer=args.rnn_layer,
                        attention_head=args.attention_head,
                        attention_layer=args.attention_layer,
                        use_residual=args.use_residual,
                        input_size=args.input_size,
                        input_in_size=args.input_in_size,
                        feature_mode=args.feature_mode,
                        n_classes=n_classes,
                        dropout=args.dropout,
                        cuda_flag=args.no_cuda)

    print ('MELD My Model.')

    if cuda:
        model.cuda()

    if args.classify == 'emotion':
        if args.class_weight:
            if args.mu > 0:
                loss_weights = torch.FloatTensor(create_class_weight(args.mu))
            else:   
                loss_weights = torch.FloatTensor([0.30427062, 1.19699616, 5.47007183, 1.95437696, 0.84847735, 5.42461417, 1.21859721])
            loss_function  = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
            loss_function_cl = CurrLearnLoss(loss_weights, delt_epoch=args.delt_epoch, sigma=args.sigma,dataset='MELD')
        else:
            if args.mu > 0:
                loss_weights = torch.FloatTensor(create_class_weight(args.mu))
            else:   
                loss_weights = torch.FloatTensor([0.30427062, 1.19699616, 5.47007183, 1.95437696, 0.84847735, 5.42461417, 1.21859721])
            loss_function = MaskedNLLLoss()
            loss_function_cl = CurrLearnLoss(loss_weights, delt_epoch=args.delt_epoch, sigma=args.sigma,dataset='MELD')
            
    else:
        loss_function  = MaskedNLLLoss()
        loss_function_cl = CurrLearnLoss(delt_epoch=args.delt_epoch, sigma=args.sigma,dataset='MELD')

    if args.loss_type=='auto_loss':
        awl = AutomaticWeightedLoss(2)
        optimizer = optim.AdamW([
                    {'params': model.parameters()},
                    {'params': awl.parameters(), 'weight_decay': 0}], lr=args.lr, weight_decay=args.l2, amsgrad=True)
    else:    
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2, amsgrad=True)

    train_loader, valid_loader, test_loader = get_MELD_loaders(path=args.data_path,
                                                               batch_size=batch_size, 
                                                               classify=args.classify,
                                                               num_workers=0)

    valid_losses, valid_fscores = [], []
    test_fscores, test_losses = [], []
    best_loss, best_label, best_pred, best_mask = None, None, None, None
    best_fscore = []
    best_acc = None

    for e in range(n_epochs):
        start_time = time.time()
        train_loss, train_acc, _, _, _, train_fscore, _ = train_or_eval_model(model, loss_function, loss_function_cl, train_loader, e, optimizer, train=True, cuda_flag=cuda,loss_type=args.loss_type,lambd=args.lambd,epochs=args.epochs)
        valid_loss, valid_acc, _, _, _, valid_fscore, _ = train_or_eval_model(model, loss_function, loss_function_cl, valid_loader, e, cuda_flag=cuda,loss_type=args.loss_type,lambd=args.lambd,epochs=args.epochs)
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, test_vids = train_or_eval_model(model, loss_function, loss_function_cl, test_loader, e, cuda_flag=cuda,loss_type=args.loss_type,lambd=args.lambd,epochs=args.epochs)
            
        valid_losses.append(valid_loss)
        valid_fscores.append(valid_fscore)
        test_losses.append(test_loss)
        test_fscores.append(test_fscore)       
        
        if args.tensorboard:
            writer.add_scalar('test: accuracy/loss', test_acc/test_loss, e)
            writer.add_scalar('train: accuracy/loss', train_acc/train_loss, e)
        
        x = 'epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'.format(e+1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_acc, test_fscore, round(time.time()-start_time, 2))
        
        print(x)

        if best_fscore == [] or best_fscore[0] < test_fscore[0]: 
            best_fscore, best_acc = test_fscore, test_acc
            best_label, best_pred, best_mask = test_label, test_pred, test_mask

        if (e+1)%10 == 0:
                x2 = classification_report(best_label, best_pred,sample_weight=best_mask, digits=4, zero_division=0)
                print(x2)
                
                np.set_printoptions(suppress=True)
                x3 = confusion_matrix(best_label, best_pred, sample_weight=best_mask)
                print(x3)

                x4 = 'test_best_acc: {}, [test_best_fscore]: {}'.format(best_acc, best_fscore)
                print(x4)

                print('-'*150)

    if args.tensorboard:
        writer.close()
        
    test_fscores = np.array(test_fscores).transpose()

    test_best_fscore = np.max(test_fscores[0])
    test_best_micro = np.max(test_fscores[1])

    print('test_best_fscore:', test_best_fscore, test_best_micro)

    scores = [test_best_fscore, test_best_micro]
    scores = [str(item) for item in scores]

    if args.classify == 'emotion':
        rf = open('results/meld_emotion_results.txt', 'a')
    elif args.classify == 'sentiment':
        rf = open('results/meld_sentiment_results.txt', 'a')
    
    rf.write('\t'.join(scores) + '\t' + str(args) + '\n')
    rf.close()
    
