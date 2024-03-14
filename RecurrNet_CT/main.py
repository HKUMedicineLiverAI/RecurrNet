import os
import torch
import torch.optim as optim

# from networks import DeepSurvCNN
# # from networks import AlexNet
# from networks import resnet152_cbam_p23
# # from networks import NegativeLogLikelihood
from datasets import SurvivalDataset
from datasets import SurvivalDataset_mask
# from transformer import ResNet
# from transformer import ViT
# from transformer import Transformer3D

from utils import c_index
from utils import NLog_Likelihood
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as TF
import random
import numpy as np
from sksurv.metrics import cumulative_dynamic_auc
# from segall import segallp23
from RecurrNET_CT  import RecurrNET_CT_p23
# from unetr.unetr import UneTR
import pandas as pd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def transform_train(mask):
    # print(mask.shape)
    mask = np.transpose(mask,(2,0,1))
    mask = [TF.to_pil_image(mask[i]) for i in range(32)]
    # # resize
    # mask = [TF.resize(mask[i],[128,128]) for i in range(32)]
    # # Random horizontal flipping
    # if random.random() > 0.5:
    #     mask = [TF.hflip(mask[i]) for i in range(32)]
    # # Random vertical flipping
    # if random.random() > 0.5:
    #     mask = [TF.vflip(mask[i]) for i in range(32)]
    # # rotate
    if random.random() > 0.8:
        mask = [TF.rotate(mask[i],15) for i in range(32)]
    # # Random affine
    # affine_param = transforms.RandomAffine.get_params(
    #     degrees = [-180, 180], translate = [0.3,0.3],  
    #     img_size = [img_w, img_h], scale_ranges = [1, 1.3], 
    #     shears = [2,2])
    # image = TF.affine(image, 
    #                   affine_param[0], affine_param[1],
    #                   affine_param[2], affine_param[3])
    # mask = TF.affine(mask, 
    #                  affine_param[0], affine_param[1],
    #                  affine_param[2], affine_param[3])
    mask = np.array([np.array(mask[i]) for i in range(32)])
    # mask = np.transpose(mask,(1,2,0))
    # # Randome GaussianBlur -- only for images
    # if random.random() < 0.1:
    #     sigma_param = random.uniform(0.01, 1)
    #     image = gaussian(image, sigma=sigma_param)
    # # Randome Gaussian Noise -- only for images
    # if random.random() < 0.1:
    #     factor_param = random.uniform(0.01, 0.5)
    #     image = image + factor_param * image.std() * np.random.randn(image.shape[0], image.shape[1])
    # # Unsharp filter -- only for images
    # if random.random() < 0.1:
    #     radius_param = random.uniform(0, 5)
    #     amount_param = random.uniform(0.5, 2)
    #     image = unsharp_mask(image, radius = radius_param, amount=amount_param)
    # mask = TF.to_tensor(mask)
    # mask = mask*255
    # normalize
    # image = TF.normalize(image,[0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # print(mask.shape)
    return mask

def transform_test(mask):
    mask = np.transpose(mask,(2,0,1))
    mask = [TF.to_pil_image(mask[i]) for i in range(32)]
    mask = np.array([np.array(mask[i]) for i in range(32)])
    # mask = TF.to_tensor(mask)
    # print(mask.shape)
    return mask

def train():
    epochs = 100
    # optim = 'Adam'
    datapath = './data/'
    model = RecurrNET_CT_p23().to(device)
    criterion = NLog_Likelihood(0.1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.7)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_dataset = SurvivalDataset(datapath, name = 'Train', transform=transform_train)
    test_dataset = SurvivalDataset(datapath, name = 'Test', transform=transform_test)
    val_dataset = SurvivalDataset(datapath, name = 'Val', transform=transform_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True,batch_size=70)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=70)
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, batch_size=70)
    
    # training
    best_c_index = 0
    best_cph_auc = 0
    flag = 0
    for epoch in range(1, epochs+1):
        model.train()
        rloss = 0
        ta_c = 0
        for index, (X2, X3, y, e, pid) in enumerate(train_loader, 1):
            # makes predictions
            X2 = X2.to(device).float()
            X3 = X3.to(device).float()
            y = y.to(device)
            e = e.to(device)
            risk_pred = model(X2,X3)
            # print(risk_pred)
            if index == 1:
                trp = risk_pred
                train_id = pid
            else:
                trp = torch.cat((trp,risk_pred),0)
                train_id = train_id + pid

            neg_log_loss, l2_loss = criterion(risk_pred, y, e, model)
            train_c = c_index(-risk_pred, y, e)
            train_loss = neg_log_loss
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            ta_c = ta_c + train_c
            rloss = rloss + train_loss.item()
            print('\nepoch %3d/%3d batch %3d/%3d train loss: %5.3f train neg loss: %5.3f train c-index: %5.4f' % (epoch, epochs, index, len(train_loader),rloss/index,neg_log_loss,ta_c/index), end='')
        scheduler.step()
        ta_c = ta_c/len(train_loader)

        # test
        model.eval()
        with torch.no_grad():
            rloss = 0
            te_c = 0
            for index, (X2, X3,y, e, pid) in enumerate(test_loader, 1):
                X2 = X2.to(device).float()
                X3 = X3.to(device).float()
                y = y.to(device)
                e = e.to(device)
                trisk_pred = model(X2,X3)
                if index == 1:
                    tp = trisk_pred
                    test_id = pid
                else:
                    tp = torch.cat((tp,trisk_pred),0)
                    test_id = test_id + pid
                neg_log_loss, l2_loss = criterion(trisk_pred, y, e, model)
                test_c = c_index(-trisk_pred, y, e)
                valid_loss = neg_log_loss
                te_c = te_c + test_c
                rloss = rloss + valid_loss.item()
                print('\nepoch %3d/%3d batch %3d/%3d test loss: %5.3f test neg loss: %5.3f test c-index: %5.4f' % (epoch, epochs, index, len(test_loader),rloss/index,neg_log_loss,te_c/index), end='')
        te_c = te_c/len(test_loader)

        # val
        model.eval()
        with torch.no_grad():
            rloss = 0
            v_c = 0
            for index, (X2, X3,y, e, pid) in enumerate(val_loader, 1):
                X2 = X2.to(device).float()
                X3 = X3.to(device).float()
                y = y.to(device)
                e = e.to(device)
                vrisk_pred = model(X2,X3)
                if index == 1:
                    vp = vrisk_pred
                    val_id = pid
                else:
                    vp = torch.cat((vp,vrisk_pred),0)
                    val_id = val_id + pid
                neg_log_loss, l2_loss = criterion(vrisk_pred, y, e, model)
                valid_c = c_index(-vrisk_pred, y, e)
                valid_loss = neg_log_loss
                v_c = v_c + valid_c
                rloss = rloss + valid_loss.item()
                print('\nepoch %3d/%3d batch %3d/%3d val loss: %5.3f val neg loss: %5.3f val c-index: %5.4f' % (epoch, epochs, index, len(val_loader),rloss/index,neg_log_loss,v_c/index), end='')
        v_c = v_c/len(val_loader)

        if best_c_index < te_c:
            best_c_index = te_c
            val_c_index = v_c

            cph_auc_train, cph_mean_auc_train = cumulative_dynamic_auc(
                    train_dataset.get_y(), train_dataset.get_y(), 
                    np.squeeze(trp.detach().cpu().numpy()), np.arange(1, 7))
            
            cph_auc, cph_mean_auc = cumulative_dynamic_auc(
                    train_dataset.get_y(), test_dataset.get_y(), 
                    np.squeeze(tp.detach().cpu().numpy()), np.arange(1, 7))

            cph_auc_val, cph_mean_auc_val = cumulative_dynamic_auc(
                    train_dataset.get_y(), val_dataset.get_y(), 
                    np.squeeze(vp.detach().cpu().numpy()), np.arange(1, 7))

            print()
            print(cph_auc_train)
            print(cph_auc)
            print(cph_auc_val)
            flag = 0
            # saves the best model
            torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch}, os.path.join(models_dir, 'best.pth'))

            train_id = list(train_id)
            trp = [k[0] for k in trp.cpu().detach().numpy().tolist()]
            test_id = list(test_id)
            tp = [k[0] for k in tp.cpu().detach().numpy().tolist()]
            val_id = list(val_id)
            vp = [k[0] for k in vp.cpu().detach().numpy().tolist()]
            pid = train_id + test_id + val_id
            risk_score = trp + tp + vp
            rs = pd.DataFrame({'Pseudo ID':pid, 'rs': risk_score})
            rs.to_csv("rs.csv",header=True)
            # print(rs)




        else:
            flag += 1
            if flag >= 25:
                return best_c_index, cph_auc, val_c_index, cph_auc_val
    
    return best_c_index, cph_auc, val_c_index, cph_auc_val




if __name__ == '__main__':
    models_dir = './model/'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    best_c_index, cph_auc, val_c_index, cph_auc_val = train()
    print()
    print(best_c_index, cph_auc)
    print(val_c_index, cph_auc_val)

