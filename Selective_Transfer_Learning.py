import torch
import torch.nn as nn
from Model_utilities import DepthWiseConv

## Here we will learn how to set the weights of a new model from a pretrained model.


## Assume that this "FirstModel" was our trained model
## and it consisted of 4 depthwise separable convolutions followied by two-activation 
## functions.
FirstModel = nn.Sequential(DepthWiseConv(32,64,stride = 1),
                         DepthWiseConv(64,128,stride =2),
                         DepthWiseConv(128,128,stride =1),
                         DepthWiseConv(128,256,stride =2),
                         nn.PReLU() ,
                         nn.ReLU())  


data = torch.randint(0,10,size = (1,32,112,112))



### Now I want to create one more model called the "SecondModel" which has just two 
## depthwise convs and one activation fucntion.
SecondModel = nn.Sequential(DepthWiseConv(32,64,stride = 1),
                    DepthWiseConv(64,128,stride =2),
                    nn.ReLU())



### MY GOAL IS TO TRANSFER THE WEIGHTS OF FIRST 2 DW-CONV WEIGHTS IN THE FIRST-MODEL 
### TO THE 2 DW-CONV WEIGHTS IN SECOND-MODEL. 
## Technically we can do this, because the channel and filter size are same.


### Before jumping right to code, lets explore a method "state_dict()"of an nn.Module

# print(SecondModel.state_dict().items())

## If you print this statement, you will get a dictionary of layer-names along with the weights
## associated with it.

## Notice that, the keys present in FirstModel will be same as in SecondModel,
## except that their weight-shape can match or dont match.
## Concretely, if we have two models and we know they both appear at a certain level
## with same weight-shape, we can transer their weights. 
## ITS JUST LIKE UPDATING A DICTIONARY.


# Please consider the example below.
d1 = {"l1":"one","l2":"two","l3":"three","l4":"four"}
d2 = {"l1":"neo","l2":"tow","l3":"ienn"}

## The above dictionarys have same keys which overlap, and there are some corresponding values associated to them,
## which are same but their len(value) doent match.

## SO WHAT IF I wanted to [ update or get new dictionary ] from the values of d2 with values in d1 ?

d2.update({k:d1[k] for k,v in d2.items() if k in d1 and len(v)==len(d1[k])})

print("d2 is ",d2)
## Notice that now "d2" dictionary has key-value pairs from the first dict "d1".


### Lets apply the same logic to our FirstModel and SecondModel.

weights1 = FirstModel.state_dict()
weights2 = SecondModel.state_dict()

weights2.update({k:weights1[k] for k,v in weights2.items() if k in weights1})
## Now this dict "weights2" has the weights from the previous model.

# Lets load these weights insde our SecondModel now. Its just one-line.
SecondModel.load_state_dict(weights2)

## The second model has now been equipped with some learnt-weights from the FirstModel. 


## Lets verify if the weights have been set correctly for a sanity check.


params_from_first  = []
params_from_second = []

layer_names = [layer_name for layer_name, _ in SecondModel.state_dict().items() if layer_name in FirstModel.state_dict()]


for layer_name, params in FirstModel.state_dict().items():
	if layer_name in layer_names:
		params_from_first.append(params)

for layer_name, params in SecondModel.state_dict().items():
	if layer_name in layer_names:
		params_from_second.append(params)



print("Have the weights been transferred ? ",all([torch.equal(params_from_second[i],params_from_first[i]) for i in range(len(params_from_first))]))

if __name__ == '__main__':
	op = FirstModel(data)
	# print(op.shape)
	# print(d2)
	# print(weights2)

	op2 = SecondModel(data)
	print(op2.shape)

	# print(names)