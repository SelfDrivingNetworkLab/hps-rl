from Steed.optimizationfns import MultiClassLM
import numpy as np
import torch
import torch.nn
CUDA=True
device = torch.device("cuda" if CUDA else "cpu")
x = [1,3,5]
x_tensor = torch.FloatTensor(x).to(device)
y = [[1],[3],[1]]
y_tensor = torch.LongTensor(y).to(device)
model = torch.nn.Sequential(
    torch.nn.Linear(1, 3,bias=False),
    torch.nn.ReLU(),
    torch.nn.Linear(3,1,bias=False),
    torch.nn.Softmax()
)
model.to(device)
lmModel = MultiClassLM.LM(model.parameters(),input=x_tensor,target=y_tensor,model=model)
for i in range(10):
   lossValue = lmModel.step()
   print(lossValue)
