���α�������Ҫ���Բο���/disk5/zhangdong/code/tiramisu_dc_b  ��ԭʾ�����룺/disk5/zhangdong/code/deformable-convolution-pytorch

�÷���
	1.������Ҫ��Դ�뿽����Ŀ��Ŀ¼�£�����ʾ���������readme���б��룻
	2. Ŀǰ���α����ѱ���װ��һ����( /disk5/zhangdong/code/tiramisu_dc_b/defcon.py) 
	����ͨpytorch��һ����init�����ʼ������������ʱ������ز������ɣ����Բ���/disk5/zhangdong/code/tiramisu_dc_b/��
	�÷�ʾ�����£�
	########################
		 # self.add_module('offconv', nn.Conv2d(in_channels=in_channels,
        #                                      out_channels=2 * 3 * 3 * self.num_deformable_groups, kernel_size=(3,3),
        #                                      stride=(1,1),padding=(1,1), bias = False).cuda())
        # self.add_module('deconv', ConvOffset2d(in_channels=in_channels,
        #                                        out_channels=in_channels, kernel_size=(3,3), stride=1,
        #                                        padding=1, num_deformable_groups=self.num_deformable_groups).cuda())
	#################################
	3.ͨ����ͨ������offset��Ȼ�����deconv�õ����α��������