import torch
import torch.nn as nn



class LinearNN(nn.Module):
    def __init__(self):
        super().__init__()
        # self.need_bias = need_bias
        self.l1 = nn.Linear(15,10,bias = True)
        self.l2 = nn.Linear(10,7,bias = True)
        self.l3 = nn.Linear(7,3,bias = True)
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x


def huk_fw(model,ip,o):
    temp_ip = ip[0].clone()
    temp_op = None
    print("Actual ip and op shape are ",ip[0].shape, o.shape,"\n")
    for name, mod in model.named_children():
        print("Temp ip shape is ",temp_ip.shape)
        temp_op = mod(temp_ip)
        print(name,mod.weight.shape,mod.bias.shape)
        print("Temp op shape is ",temp_op.shape,"\n")
        temp_ip  = temp_op
    return


model = LinearNN()
model.register_forward_hook(huk_fw)

num = 1
Inp = torch.randint(0,10,size = (num,15))
opt = model(Inp)
L =10-opt.sum()
L.backward()




