import torch
import numpy as np
from math import log10,sqrt
import cv2 as cv
from model import make_ARCNN_model,make_RDN_model,make_SRCNN_model,make_VDSR_model
from model_upsample import  make_RDN_model_upsample,make_ARCNN_model_upsample
from test_Img import SR_Separate
from test_Img_7_14 import SR_Separate_25
import torch.backends.cudnn as cudnn
import gc
import os

from sewar.full_ref import ssim,psnr

import matplotlib.pyplot as plt

class Trainer(object):
    def __init__(self, config, training_loader,testing_loader):
        super(Trainer, self).__init__()
        self.GPU_IN_USE = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.GPU_IN_USE else 'cpu')
        self.model_1 = None
        self.model_2 = None
        self.model_3 = None
        self.model_4 = None
        self.lr = config.lr
        self.nEpochs = config.nEpochs
        self.criterion = None
        self.optimizer = None
        self.scheduler = None      #lr 下降方式的控制
        self.seed = config.seed
        self.upscale_factor = config.upscale_factor
        self.training_loader = training_loader
        self.testing_loader = testing_loader
        self.fig = config.fig

    def build_model(self,args):
        # self.model_1 = make_RDN_model(args,'SAI').to(self.device)
        # self.model_2 = make_RDN_model(args,'EPI_1').to(self.device)
        # self.model_3 = make_RDN_model(args,'EPI_2').to(self.device)

        self.model_1 = make_RDN_model_upsample(args,'SAI').to(self.device)
        self.model_2 = make_RDN_model_upsample(args,'EPI_1').to(self.device)
        self.model_3 = make_RDN_model_upsample(args,'EPI_2').to(self.device)

        self.model_4 = make_ARCNN_model_upsample().to(self.device)
        self.criterion = torch.nn.MSELoss()
        torch.manual_seed(self.seed)

        if self.GPU_IN_USE:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.criterion.cuda()

        self.optimizer = torch.optim.Adam([
            {'params': self.model_1.parameters()},
            {'params': self.model_2.parameters()},
            {'params': self.model_3.parameters()},
            {'params': self.model_4.parameters()}],lr=self.lr, betas=(0.9, 0.999), eps=1e-8)

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[30, 50, 60], gamma=0.5)  # lr decay

    def save(self):
        model_out_path = "EPI_model_upFrame_4_8_C_SF_upsamle.pth"
        state = {'model_1': self.model_1.state_dict(),
                 'model_2': self.model_2.state_dict(),
                 'model_3': self.model_3.state_dict(),
                 'model_4': self.model_4.state_dict()}
        torch.save(state, model_out_path)
        # torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def train(self,epoch):
        self.model_1.train()
        self.model_2.train()
        self.model_3.train()
        self.model_4.train()

        f = open("loss_upFrame_4_8_C_SF_upsample.txt", "a")

        train_loss = 0

        for batch_num, (data, target) in enumerate(self.training_loader):
            if batch_num >= 1000:
                break
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()

            # data = data.permute([0,2,3,1])  matlab写成的H5需要
            # target = target.permute([0, 2, 3, 1])

            LF1 = self.model_1(data)
            LF1 = LF1.squeeze()
            LF1 = LF1.permute([0,2,3,1])
            loss_1 = self.criterion(LF1, target)

            ##w,n方向
            LF2 = self.model_2(data)
            LF2 = LF2.squeeze()
            LF2 = LF2.permute([0, 2, 3, 1])
            loss_2 = self.criterion(LF2, target)

            ##h,n方向
            LF3 = self.model_3(data)
            LF3 = LF3.squeeze()
            LF3 = LF3.permute([0, 2, 3, 1])
            loss_3 = self.criterion(LF3, target)


            LF = torch.mean(torch.stack([LF1, LF2, LF3]), 0).cuda()

            #Enhance NEt
            prediction = self.model_4(LF)
            prediction = prediction.squeeze()
            loss_4 = self.criterion(prediction, target)

            loss = loss_1+loss_2+loss_3+loss_4
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            print(" Epoch:{:d}\tbatch_num:{:d}\tLoss: {:.4f}\t".format(epoch,batch_num + 1, train_loss / (batch_num + 1)))

        # print("    Average Loss: {:.4f}".format(train_loss / len(self.training_loader)))
        print("=========Epoch:{:d}\t Average Loss: {:.4f}".format(epoch,train_loss/ 1000))
        f.write("Epoch:{:d},  Average Loss: {:.4f}\n".format(epoch,train_loss / 1000))
        f.close()

    def test(self):

        model_out_path = "EPI_model_upFrame.pth"
        checkpoint = torch.load(model_out_path)
        self.model_1.load_state_dict(checkpoint['model_1'])
        self.model_2.load_state_dict(checkpoint['model_2'])
        self.model_3.load_state_dict(checkpoint['model_3'])
        self.model_4.load_state_dict(checkpoint['model_4'])
        self.model_1.eval()
        self.model_2.eval()
        self.model_3.eval()
        self.model_4.eval()

        f = open("Average psnr_Image_upFrame_4_4_8_8.txt", "a")
        avg_psnr = 0
        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.testing_loader):
                data, target = data.to(self.device), target.to(self.device)

                LF1 = self.model_1(data)
                LF1 = LF1.squeeze(1)
                LF1 = LF1.permute([0, 2, 3, 1])

                ##w,n方向
                LF2 = self.model_2(data)
                LF2 = LF2.squeeze(1)
                LF2 = LF2.permute([0, 2, 3, 1])

                ##h,n方向
                LF3 = self.model_3(data)
                LF3 = LF3.squeeze(1)
                LF3 = LF3.permute([0, 2, 3, 1])

                LF = torch.mean(torch.stack([LF1, LF2, LF3]), 0).cuda()

                # Enhance NEt
                prediction = self.model_4(LF)
                prediction = prediction.squeeze(1)

                psnr_score = np.zeros((1, 64))
                for i in range(64):
                    psnr_score[:,i]  = Cp_PSNR(prediction[0,:,:,i],target[0,:,:,i])
                    # print("  PSNR: {:.4f}".format(psnr_score[:,i] ))
                PSNR = np.mean(psnr_score)
                print("  PSNR: {:.4f}".format(PSNR))
                f.write("PSNR: {:.4f}\n".format(PSNR))
                avg_psnr += PSNR
            print("=========Average Psnr: {:.4f}".format(avg_psnr/ len(self.testing_loader)))
            f.write("=========Average Psnr: {:.4f}\n".format(avg_psnr/ len(self.testing_loader)))
        f.close()

    def test_img(self):
        f = open("Average psnr_ssim_Image_upFrame_EPFL_7_14.txt", "a")
        a = 14
        avg_psnr = 0
        avg_ssim = 0
        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.testing_loader):
                print("####################{:d}############".format(batch_num))
                data, target = data.to(self.device), target.to(self.device)

                savepath = "./Save_Img_EPFL_7_14/"+str(batch_num)+"/"
                isExists = os.path.exists(savepath)
                if not isExists:
                   os.makedirs(savepath)



                # prediction = SR_Separate(self,data)
                prediction = SR_Separate_25(self, data)


                psnr_score = np.zeros(a*a)
                curSSIM = np.zeros(a*a)


                for i in range(a*a):

                    path = savepath + str(i)+".png"
                    img = prediction[0, :, :, i]*255
                    cv.imwrite(path, img.cpu().numpy())

                    img1 = prediction[0, :, :, i]
                    img2 = target[0, :, :, i]
                    # psnr_score[i] = Cp_PSNR(prediction[0, :, :, i], target[0,  :, :, i])
                    psnr_score[i] = psnr(img2.cpu().numpy(),img1.cpu().numpy(),MAX=1)

                    curSSIM[i] = ssim(img2.cpu().numpy(),img1.cpu().numpy(),MAX=1)[0]

                    print("psnr:{:.4f}".format(psnr_score[i]))
                    print("ssim:{:.4f}".format(curSSIM[i]))
                PSNR = np.mean(psnr_score)
                print(" each one_Img PSNR: {:.4f}".format(PSNR))
                f.write("PSNR: {:.4f}\n".format(PSNR))
                avg_psnr += PSNR

                SSIM = np.mean(curSSIM)
                print(" each one_Img SSIM: {:.4f}".format(SSIM))
                f.write("SSIM: {:.4f}\n".format(SSIM))
                avg_ssim += SSIM

                del data, target,prediction
                gc.collect()

            print("=========Average Psnr: {:.4f}".format(avg_psnr / len(self.testing_loader)))
            f.write("=========Average Psnr: {:.4f}\n".format(avg_psnr / len(self.testing_loader)))

            print("=========Average SSIM: {:.4f}".format(avg_ssim / len(self.testing_loader)))
            f.write("=========Average SSIM: {:.4f}\n".format(avg_ssim / len(self.testing_loader)))
        f.close()

    def run(self,args):

        if self.fig =='train':
            self.build_model(args)

            for epoch in range(1, self.nEpochs + 1):
                print("\n===> Epoch {} starts:".format(epoch))

                self.train(epoch)
                self.scheduler.step(epoch)
                if epoch % 1 == 0:
                    self.save()
        if self.fig == 'test':
            self.build_model(args)
            # self.test()
            self.test_img()

def Cp_PSNR(img1,img2):


    diff = img1 - img2

    # plt.figure(3)
    # plt.imshow(diff.cpu(), cmap='gray')

    diff = diff.flatten()
    rmse = sqrt(torch.mean(diff ** 2.))
    return 20 * log10(1.0 / rmse)

    # mse = np.mean((img1  - img2 ) ** 2)
    # if mse < 1.0e-10:
    #     return 100
    # PIXEL_MAX = 1
    # return 20 * log10(PIXEL_MAX / sqrt(mse))
