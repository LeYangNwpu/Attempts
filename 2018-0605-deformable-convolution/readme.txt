可形变卷积，主要可以参考：/disk5/zhangdong/code/tiramisu_dc_b  及原示例代码：/disk5/zhangdong/code/deformable-convolution-pytorch

用法：
	1.将所需要的源码拷贝至目标目录下，按照示例代码里的readme进行编译；
	2. 目前可形变卷积已被封装成一个类( /disk5/zhangdong/code/tiramisu_dc_b/defcon.py) 
	跟普通pytorch类一样，init输入初始化参数，调用时输入相关参数即可，可以参照/disk5/zhangdong/code/tiramisu_dc_b/；
	用法示例如下：
	########################
		 # self.add_module('offconv', nn.Conv2d(in_channels=in_channels,
        #                                      out_channels=2 * 3 * 3 * self.num_deformable_groups, kernel_size=(3,3),
        #                                      stride=(1,1),padding=(1,1), bias = False).cuda())
        # self.add_module('deconv', ConvOffset2d(in_channels=in_channels,
        #                                        out_channels=in_channels, kernel_size=(3,3), stride=1,
        #                                        padding=1, num_deformable_groups=self.num_deformable_groups).cuda())
	#################################
	3.通过普通卷积获得offset，然后带入deconv得到可形变卷积结果。