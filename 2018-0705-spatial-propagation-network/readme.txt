spn主要参考：/disk5/zhangdong/code/train_ResNet/resnet_seg.py

引导网络的最后一层输出被分成12组，即四个大方向，每个大方向又是被分成3个方向
示例代码如下：
def spnNet(self,out, y): ##out为引导网络输出，y为粗分割结果 
        Glr1 = out[:, 0:32] ###Glr1 即：G为引导网络(guidance)，lr表示方向是从左向右（left->right），1表示第一个子方向；
		         ###依次类推，Gdu2为引导网络的从下到上的小组(down -> up)
        Glr2 = out[:, 32:64]
        Glr3 = out[:, 64:96]
        sum_abs = Glr1.abs() + Glr2.abs() + Glr3.abs()
        mask_need_norm = sum_abs.ge(1)
        mask_need_norm = mask_need_norm.float()
        Glr1_norm = torch.div(Glr1, sum_abs)
        Glr2_norm = torch.div(Glr2, sum_abs)
        Glr3_norm = torch.div(Glr3, sum_abs)

        Glr1 = torch.add(-mask_need_norm, 1) * Glr1 + mask_need_norm * Glr1_norm
        Glr2 = torch.add(-mask_need_norm, 1) * Glr2 + mask_need_norm * Glr2_norm
        Glr3 = torch.add(-mask_need_norm, 1) * Glr3 + mask_need_norm * Glr3_norm

        ylr = y.cuda()
        Glr1 = Glr1.cuda()
        Glr2 = Glr2.cuda()
        Glr3 = Glr3.cuda()
        Propagator = GateRecurrent2dnoind(True, False)

        ylr = Propagator.forward(ylr, Glr1, Glr2, Glr3)

        Grl1 = out[:, 96:128];
        Grl2 = out[:, 128:160];
        Grl3 = out[:, 160:192];
        sum_abs = Grl1.abs() + Grl2.abs() + Grl3.abs()
        mask_need_norm = sum_abs.ge(1)
        mask_need_norm = mask_need_norm.float()
        Grl1_norm = torch.div(Grl1, sum_abs)
        Grl2_norm = torch.div(Grl2, sum_abs)
        Grl3_norm = torch.div(Grl3, sum_abs)

        Grl1 = torch.add(-mask_need_norm, 1) * Grl1 + mask_need_norm * Grl1_norm
        Grl2 = torch.add(-mask_need_norm, 1) * Grl2 + mask_need_norm * Grl2_norm
        Grl3 = torch.add(-mask_need_norm, 1) * Grl3 + mask_need_norm * Grl3_norm

        yrl = y.cuda()
        Grl1 = Grl1.cuda()
        Grl2 = Grl2.cuda()
        Grl3 = Grl3.cuda()
        Propagator = GateRecurrent2dnoind(False,True)

        yrl = Propagator.forward(ylr, Grl1, Grl2, Grl3)

        Gdu1 = out[:, 192:224];
        Gdu2 = out[:, 224:256];
        Gdu3 = out[:, 256:288];
        sum_abs = Gdu1.abs() + Gdu2.abs() + Gdu3.abs()
        mask_need_norm = sum_abs.ge(1)
        mask_need_norm = mask_need_norm.float()
        Gdu1_norm = torch.div(Gdu1, sum_abs)
        Gdu2_norm = torch.div(Gdu2, sum_abs)
        Gdu3_norm = torch.div(Gdu3, sum_abs)

        Gdu1 = torch.add(-mask_need_norm, 1) * Gdu1 + mask_need_norm * Gdu1_norm
        Gdu2 = torch.add(-mask_need_norm, 1) * Gdu2 + mask_need_norm * Gdu2_norm
        Gdu3 = torch.add(-mask_need_norm, 1) * Gdu3 + mask_need_norm * Gdu3_norm

        ydu = y.cuda()
        Gdu1 = Gdu1.cuda()
        Gdu2 = Gdu2.cuda()
        Gdu3 = Gdu3.cuda()
        Propagator = GateRecurrent2dnoind(False, False)

        ydu = Propagator.forward(ydu, Gdu1, Gdu2, Gdu3)

        Gud1 = out[:, 288:320];
        Gud2 = out[:, 320:352];
        Gud3 = out[:, 352:384];
        sum_abs = Gud1.abs() + Gud2.abs() + Gud3.abs()
        mask_need_norm = sum_abs.ge(1)
        mask_need_norm = mask_need_norm.float()
        Gud1_norm = torch.div(Gud1, sum_abs)
        Gud2_norm = torch.div(Gud2, sum_abs)
        Gud3_norm = torch.div(Gud3, sum_abs)

        Gud1 = torch.add(-mask_need_norm, 1) * Gud1 + mask_need_norm * Gud1_norm
        Gud2 = torch.add(-mask_need_norm, 1) * Gud2 + mask_need_norm * Gud2_norm
        Gud3 = torch.add(-mask_need_norm, 1) * Gud3 + mask_need_norm * Gud3_norm

        yud = y.cuda()
        Gud1 = Gud1.cuda()
        Gud2 = Gud2.cuda()
        Gud3 = Gud3.cuda()
        Propagator = GateRecurrent2dnoind(True, True)

        yud = Propagator.forward(yud, Gud1, Gud2, Gud3)

        yout = torch.max(ylr,yrl)
        yout = torch.max(yout,yud)
        yout = torch.max(yout,ydu)
        #print("yout size", yout.size())
        #print("Gud1 size",Glr1.size())
        return yout
