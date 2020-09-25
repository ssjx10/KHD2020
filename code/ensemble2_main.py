import os
import argparse
import sys
import time
import cv2 
import numpy as np
import random
import nsml
from nsml.constants import DATASET_PATH, GPU_NUM
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.transforms import Compose, RandomCrop, Pad, RandomHorizontalFlip, RandomVerticalFlip, RandomResizedCrop, Resize
from torchvision.transforms import ToPILImage, ToTensor, Normalize
from torch.utils.data import Subset
from PIL.Image import BICUBIC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ExponentialLR
from sklearn import metrics
from itertools import chain
from customEval import *

input_size = 224
class Ensemble(nn.Module):

    def __init__(self, model_list):
        super().__init__()
        self.nums = len(model_list)
        self.model_list = nn.ModuleList(model_list)

    def forward(self, x):
        
        return [self.model_list[i](x[i]) for i in range(self.nums)]
    
######################## DONOTCHANGE ###########################
def bind_model(model):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        torch.save(model.state_dict(),os.path.join(dir_name, 'model'))
        print('model saved!')

    def load(dir_name):
        model.load_state_dict(torch.load(os.path.join(dir_name, 'model')))
        model.eval()
        print('model loaded!')

    def infer(image_path):
        result = []
        with torch.no_grad():             
            batch_loader = DataLoader(dataset=PathDataset(image_path, labels=None, input_size=input_size),
                                        batch_size=batch_size,shuffle=False)
            # Train the model 
            for i, images_list in enumerate(batch_loader):
                y_hat_list = model([images_list[j].to(device) for j in range(len(images_list))])
                y_hat = [torch.softmax(y_hat_list[j], 1).data.cpu().numpy() for j in range(len(images_list))]
                y_hat = np.mean(y_hat, 0)
                result.extend(np.argmax(y_hat, axis=1))

        print('predicted')
        return np.array(result)

    nsml.bind(save=save, load=load, infer=infer)


def path_loader (root_path):
    image_path = []
    image_keys = []
    for _,_,files in os.walk(os.path.join(root_path,'train_data')):
        for f in files:
            path = os.path.join(root_path,'train_data',f)
            if path.endswith('.png'):
                image_keys.append(int(f[:-4]))
                image_path.append(path)

    return np.array(image_keys), np.array(image_path)


def label_loader (root_path, keys):
    labels_dict = {}
    labels = []
    with open (os.path.join(root_path,'train_label'), 'rt') as f :
        for row in f:
            row = row.split()
            labels_dict[int(row[0])] = (int(row[1]))
    for key in keys:
        labels = [labels_dict[x] for x in keys]
    return labels
############################################################

# make a transform
def make_transform(input_size, mode='train', flag=0):
    if flag == 2:
        train_transform = Compose([
            ToPILImage(),
    #         RandomResizedCrop(input_size),
            Resize(input_size, BICUBIC),
    #         RandomCrop(input_size),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif flag == 0:
        train_transform = Compose([
            ToPILImage(),
            Resize(input_size, BICUBIC),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif flag == 1:
        train_transform = Compose([
            ToPILImage(),
            Resize(input_size, BICUBIC),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    test_transform = Compose([
        ToPILImage(),
        Resize(input_size, BICUBIC),    
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if mode == 'train':
        return train_transform
    
    return test_transform
    

def initialize_model(model_name, num_classes, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model = models.resnet50(pretrained=use_pretrained)
        num_ftrs = model.fc.in_features
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299
        
    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model = models.inception_v3(pretrained=use_pretrained)
        # Handle the auxilary net
        num_ftrs = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299
        
    return model, input_size

class PathDataset(Dataset): 
    def __init__(self,image_path, labels=None, test_mode= True, mode='test', input_size=[224,224,224]): 
        self.len = len(image_path)
        self.image_path = image_path
        self.labels = labels 
        self.mode = test_mode
        self.nums = len(input_size)
        
        self.transform = []
        for i in range(self.nums):
            self.transform.append(make_transform(input_size[i], mode, i))

    def __getitem__(self, index): 
        im = cv2.imread(self.image_path[index])
        
                ### REQUIRED: PREPROCESSING ###
        im_list = [torch.tensor(self.transform[i](im)) for i in range(self.nums)]

        if self.mode:
            return im_list
        else:
            return im_list,\
                 torch.tensor(self.labels[index], dtype=torch.long)

    def __len__(self): 
        return self.len
    
def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        
def sensi_speci(y_true, y_pred):
    # y_pred shape : (n,)
    pred = y_pred > 0.5
    pred = pred.reshape(-1).astype(np.uint8)

    zero_one =0
    zero_zero =0
    one_zero =0
    one_one =0
    # one -> true
    for tr, pr in zip(y_true.astype(np.uint8), pred):
        tr = str(tr).strip()
        pr = str(pr).strip()
        if pr == '0':
            if tr == '0':
                zero_zero +=1
            elif tr =='1':
                zero_one +=1
        elif pr == '1':
            if tr == '0':
                one_zero +=1
            elif tr =='1':
                one_one +=1

    try: specificity = zero_zero / (zero_zero + one_zero)
    except: specificity=0
    try: sensitivity = one_one / (zero_one + one_one)    
    except: sensitivity=0
    try: acc = (one_one + zero_zero) / (zero_zero + one_zero + zero_one + one_one)    
    except: acc = 0

    return acc, sensitivity, specificity

def KLDiv(pred, targets, T=1):
    logsoftmax = nn.LogSoftmax(dim=1)
    softmax = nn.Softmax(dim=1)
    soft_targets = softmax(targets/T)
    return nn.KLDivLoss()(logsoftmax(pred), soft_targets)

if __name__ == '__main__':

    ########## ENVIRONMENT SETUP ############
    args = argparse.ArgumentParser()

    ########### DONOTCHANGE: They are reserved for nsml ###################
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')
    ######################################################################

    # hyperparameters
    args.add_argument('--epoch', type=int, default=50)
    args.add_argument('--batch_size', type=int, default=32) 
    args.add_argument('--learning_rate', type=int, default=0.001)

    config = args.parse_args()
    
    # set the seed
    seed = 1993
    seed_everything(seed)
        
    # training parameters
    num_epochs = config.epoch
    batch_size = config.batch_size
    num_classes = 2
    learning_rate = config.learning_rate 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # model setting ## 반드시 이 위치에서 로드해야함
    print(f'Model setting')
    model_name = 'resnet'
    is_inception = False
    if model_name == 'inception':
        is_inception = True
        
    # model load
    model_list = []
    # model1
    model1, _ = initialize_model(model_name, 2)
    model1 = model1.to(device)
#     bind_model(model1)
#     nsml.load(checkpoint='6', session='KHD005/Breast_Pathology/276')
    model_list.append(model1)
    # model2
    model2, _ = initialize_model(model_name, 2)
    model2 = model2.to(device)
#     bind_model(model2)
#     nsml.load(checkpoint='6', session='KHD005/Breast_Pathology/276')
    model_list.append(model2)
    # model3
#     model3, _ = initialize_model(model_name, 2)
#     model3 = model3.to(device)
#     bind_model(model3)
#     nsml.load(checkpoint='31', session='KHD005/Breast_Pathology/304')
#     model_list.append(model3)
    
    # ensemble model
    model = Ensemble(model_list)
    input_size = [299, 331]

    #     optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # initialize
    #     nn.init.normal_(model.fc.weight, std=0.02)
    #     nn.init.normal_(model.fc.bias, 0)

    # Loss and optimizer
    loss_ce = nn.CrossEntropyLoss()

    lr_sch = ExponentialLR(optimizer, gamma=0.975)

    ############ DONOTCHANGE ###############
    bind_model(model)
    if config.pause: ## test mode 일때는 여기만 접근
        print('Inferring Start...')
        nsml.paused(scope=locals())
    #######################################

    if config.mode == 'train': ### training mode 일때는 여기만 접근
        print('Training Start...')
        
#         nsml.load(checkpoint='31', session='KHD005/Breast_Pathology/304')
#         nsml.save('LAST')
#         exit()

        ############ DONOTCHANGE: Path loader ###############
        root_path = os.path.join(DATASET_PATH,'train')
        image_keys, image_path = path_loader(root_path)
        labels = label_loader(root_path, image_keys)
        ##############################################

        labels = np.array(labels)
        # train_valid split
        data_num = len(labels)
        indices = np.arange(data_num)
        x_train, x_val, y_train, y_val, idx1, idx2 = train_test_split(image_path, labels, indices, test_size=0.2, stratify=labels, random_state=20)

        # train_val
        print(f'Data Loading')
        use_split = False
        if use_split:
            x_train, x_val = image_path[tr_idx], image_path[vl_idx]
            y_train, y_val = labels[tr_idx], labels[vl_idx]

            train_loader = DataLoader(\
                dataset=PathDataset(x_train, y_train, test_mode=False, mode='train', input_size=input_size), 
                    batch_size=batch_size, shuffle=True, drop_last=True, num_workers=3)
            val_loader = DataLoader(\
                dataset=PathDataset(x_val, y_val, test_mode=False, mode='val', input_size=input_size), 
                    batch_size=batch_size, shuffle=False, drop_last=False, num_workers=3)
            train_num = train_loader.dataset.len
            val_num = val_loader.dataset.len
        else:
            train_loader = DataLoader(\
                dataset=PathDataset(image_path, labels, test_mode=False, mode='train', input_size=input_size), 
                    batch_size=batch_size, shuffle=True, drop_last=True, num_workers=12)
            train_num = train_loader.dataset.len

        print_every = 1
        # Train the model
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            tr_acc = 0.0
            tr_y = []
            tr_pred1 = []
            tr_pred2 = []
            tr_predE = []
            
            model.train()
            for j, (x_batch_list, y_batch) in enumerate(train_loader):
                x_batch_list = [x_batch_list[j].to(device) for j in range(len(x_batch_list))]
                y_batch = y_batch.to(device)

                # Forward pass
                if is_inception:
                    outputs, aux_outputs  = model(x_batch)
                    loss1 = loss_ce(outputs, y_batch)
                    loss2 = loss_ce(aux_outputs, y_batch)
                    loss = loss1 + 0.4*loss2
                else:
                    outputs = model(x_batch_list)
                    loss = loss_ce(outputs[0], y_batch) + KLDiv(outputs[0], outputs[1]) + \
                            loss_ce(outputs[1], y_batch) + KLDiv(outputs[1], outputs[0])

                tr_y.append(y_batch.cpu().data.numpy())
                tr_pred1.append(torch.softmax(outputs[0], 1).data.cpu().numpy()[:, 1])
                tr_pred2.append(torch.softmax(outputs[1], 1).data.cpu().numpy()[:, 1])
#                     tr_pred_E.append( (torch.softmax(outputs[0], 1).data.cpu().numpy() + torch.softmax(outputs[0], 1).data.cpu().numpy())/2 )

                pred_class = np.argmax(outputs[0].data.cpu().numpy(), axis=1)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                correct = len(np.where(pred_class == y_batch.cpu().data.numpy())[0])
                tr_acc += correct

            epoch_loss /= len(train_loader)
            tr_acc /= train_num
            
            
            if use_split:
                va_loss = 0.0
                va_acc = 0.0

                va_y = []
                va_pred = []
                with torch.no_grad():
                    model.eval()
                    # Valid accuracy
                    for x_batch_list, y_batch in val_loader:

                        x_batch_list = [x_batch_list[j].to(device) for j in range(len(x_batch_list))]
                        y_batch = y_batch.to(device)

                        outputs = model(x_batch_list)
                        loss = loss_ce(outputs, y_batch)

                        va_y.append(y_batch.cpu().data.numpy())
                        va_pred.append(torch.softmax(outputs, 1).data.cpu().numpy()[:, 1])

                        pred = np.argmax(outputs.data.cpu().numpy(), axis=1)

                        va_loss += loss.item()
                        correct = len(np.where(pred == y_batch.cpu().data.numpy())[0])
                        va_acc += correct

                    va_loss /= len(val_loader)
                    va_acc /= val_num

                if (epoch + 1) % print_every == 0:
                    va_y = np.concatenate(va_y, 0)
                    va_pred = np.concatenate(va_pred, 0)
                    auc = metrics.roc_auc_score(va_y, va_pred)
                    score = evaluation_metrics(va_y, va_pred)

                    print(get_metrics(va_y, va_pred))
                    print('Epoch [{}/{}], T_Loss: {:.4f}, T_Acc: {:.4f}, V_Loss: {:.4f}, V_Acc: {:.4f}, V_AUC: {:.4f}, V_score: {:.4f}'
                            .format(epoch + 1, num_epochs, epoch_loss, tr_acc, va_loss, va_acc, auc, score))
            else:
                if (epoch + 1) % print_every == 0:
                    tr_y = np.concatenate(tr_y, 0)
                    tr_pred1 = np.concatenate(tr_pred1, 0)
                    tr_pred2 = np.concatenate(tr_pred2, 0)
                    tr_predE = (tr_pred1 + tr_pred2) / 2
                    auc = metrics.roc_auc_score(tr_y, tr_predE)
                    score1 = evaluation_metrics(tr_y, tr_pred1)
                    score2 = evaluation_metrics(tr_y, tr_pred2)
                    scoreE = evaluation_metrics(tr_y, tr_predE)

                    print(get_metrics(tr_y, tr_predE))
                    print('Epoch [{}/{}], T_Loss: {:.7f}, T_Acc: {:.7f}, T_AUC: {:.7f}, T_score1: {:.7f}, T_score2: {:.7f}, T_scoreE: {:.7f}'
                            .format(epoch + 1, num_epochs, epoch_loss, tr_acc, auc, score1, score2, scoreE))

            lr_sch.step()

            nsml.report(summary=True, step=epoch, epoch_total=num_epochs, loss=epoch_loss, v_score=scoreE)
            nsml.save('E_'+ str(epoch))
