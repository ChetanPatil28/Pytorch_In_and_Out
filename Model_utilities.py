import torch
import torch.nn as nn

class DepthWiseConv(nn.Module):
    def __init__(self,in_channels , out_channels, ksize = 3 ,stride = 1):
        super().__init__()
        self.inn = in_channels
        self.out = out_channels
        self.ksize = ksize
        self.s = stride
        self.depth_conv = nn.Conv2d(self.inn,self.inn,self.ksize,stride = self.s,padding = 1,groups = self.inn, bias = False )
        self.point_conv = nn.Conv2d(self.inn,self.out,(1,1),1,bias = False)
        self.bn1 = nn.BatchNorm2d(self.inn)
        self.bn2 = nn.BatchNorm2d(self.out)
        self.relu = nn.ReLU(inplace = True)
    def forward(self, x):
        x = self.depth_conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.point_conv(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x



# Instantiate this model.
depth_model = DepthWiseConv(32,64,ksize = 3 ,stride = 1)


## What individual modules does this depth_model consist of ?
for mod in depth_model.children():
    # print(mod)
    pass



## Setting this model's  Batch-Norm layers' bias to 7.
def set_bn_bias_to_constant(m):
    for mod in m.children():
        if type(mod)==nn.BatchNorm2d:
            mod.bias.data.fill_(7.0) 
        

depth_model.apply(set_bn_bias_to_constant)




if __name__ == '__main__':
    
    ### Lets see if the Batch_Norm Bias' have been set to '1'.!!
    for name,params in depth_model.named_parameters():
        if name.startswith("bn") and name.endswith("bias"):
            W = params[0]
            B = params[1]
            print("Batch-Norm Bias are ", B)


