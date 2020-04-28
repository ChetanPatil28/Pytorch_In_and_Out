import torch

### Consider a small-graph.
t= torch.Tensor([1])


a = torch.Tensor([5])
a.requires_grad = True
w1 = torch.Tensor([1])
w1.requires_grad = True
w2 = torch.Tensor([2])
w2.requires_grad = True
w3 = torch.Tensor([3])
w3.requires_grad = True
w4 = torch.Tensor([4])
w4.requires_grad = True
b = w1*a
c = w2*a
d = w3*b
d.register_hook(print)
e = w4*c
f = d + e
L = f -10


### Always do one operator over two tensors. 
### In that way, u will know exactly whats happening.
## For example, b = w1*a right, so its gradient will be db/dw1 = a and db/ba = w1;
## So, if u wanna see its gradient . Just do b.grad_fn(torch.Tensor([1])). Putting a 1 will make it think there is an incoming gradeint.(PyTorch is built that way)
## If you dont pass in any argument, error will be thrown.

print(b.grad_fn(t))

## db = [db/dw1, db/da]
## => db = [5,1]
 