import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def SR_Separate(self, input):
    inputLF = input
    model_out_path1 = "epoch_lr_data_RGB_video.pth"
    # model_out_path2 = "Enhanced_video_vimeoAll_4_2.pth"
    checkpoint = torch.load(model_out_path1)
    # checkpoint2 = torch.load(model_out_path2)
    self.model_1.load_state_dict(checkpoint['model_1'])
    self.model_2.load_state_dict(checkpoint['model_2'])
    self.model_3.load_state_dict(checkpoint['model_3'])
    self.model_4.load_state_dict(checkpoint['model_4'])
    self.model_5.load_state_dict(checkpoint['model_5'])
    self.model_1.eval()
    self.model_2.eval()
    self.model_3.eval()
    self.model_4.eval()
    self.model_5.eval()
    # del checkpoint,checkpoint2

    prediction1 = mode_test(self,inputLF)

    return prediction1


def mode_test(self,data):
    LF1 = self.model_1(data)
    # LF1 = LF1.squeeze(1)
    LF1 = LF1.permute([0, 3, 4, 1,2])

    ##w,n方向
    LF2 = self.model_2(data)
    # LF2 = LF2.squeeze(1)
    LF2 = LF2.permute([0, 3, 4, 1,2])

    ##h,n方向
    LF3 = self.model_3(data)
    # LF3 = LF3.squeeze(1)
    LF3 = LF3.permute([0, 3, 4, 1,2])

    LF = torch.mean(torch.stack([LF1, LF2, LF3]), 0).cuda()
    del LF1,LF2,LF3

    # Enhance NEt
    prediction = self.model_4(LF)
    prediction = prediction.permute([0, 2,3,1,4])#b,h,w,c,n
    #del LF

    # prediction = prediction.squeeze(1)
    prediction = EnhanceModel(self, prediction)

    # prediction = LF.squeeze(1)
    return prediction

def SuperResolution_Model(self,data,target):


    LF1 = self.model_1(data)#b,c,n,h,w
    # LF1 = LF1.squeeze()
    LF1 = LF1.permute([0, 3, 4, 1,2])
    # loss_1 = self.criterion(LF1, target)

    ##w,n方向
    LF2 = self.model_2(data)
    # LF2 = LF2.squeeze()
    LF2 = LF2.permute([0, 3, 4, 1,2])
    # loss_2 = self.criterion(LF2, target)

    ##h,n方向
    LF3 = self.model_3(data)
    # LF3 = LF3.squeeze()
    LF3 = LF3.permute([0, 3, 4, 1,2])
    # loss_3 = self.criterion(LF3, target)

    LF = torch.mean(torch.stack([LF1, LF2, LF3]), 0).cuda()
    loss_1 = self.criterion(LF, target)

    # Enhance NEt
    prediction = self.model_4(LF) #b,c,h,w,n
    prediction = prediction.permute([0, 2,3,1,4]) #b,h,w,c,n
    # loss_2 = self.criterion(prediction, target)


    #####EnhanceBlock
    out = EnhanceModel(self,prediction)
    loss_3 = self.criterion(out, target)
    #
    # return loss_5
    # return loss_1,loss_2,loss_3,loss_4,loss_5
    return loss_1, loss_3


def EnhanceModel(self,input):
    b, h, w, c,n = input.shape
    data_new = input
    for i in range(n):
        if i % 2 == 0 and i !=0:

            #
            data1 = input[ :, :, :, :,i - 2:i-1] #b,h,w,c,n
            data2 = input[:, :, :, :, i - 1:i ]  # b,h,w,c,n
            data3 = input[:, :, :, :, i :i + 1]  # b,h,w,c,n
            data = torch.cat((data1,data2,data3),3) #b,h,w,c
            data = data.squeeze(4)
            data = data.permute([0, 3, 1, 2 ])  #b,c,h,w
            inter = self.model_5(data)  #b,c,h,w
            inter = inter.permute([0, 2, 3, 1]) #b,h,w,c
            inter = inter.unsqueeze(4)
            data_new[:,:,:,:,i-1:i] = inter

    return data_new
