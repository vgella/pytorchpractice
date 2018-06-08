import torch
batch_size, input, hidden, output = 64,1000,100,10
x = torch.randn(batch_size, input)
y = torch.randn(batch_size, output)

model = torch.nn.Sequential(torch.nn.Linear(input, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, output),)

loss_fn = torch.nn.MSELoss(size_average = False)

learning_rate = 1e-4

for i in range(500):
   y_pred = model(x)

   loss = loss_fn(y_pred, y)
   print(loss.item(), ": %d" % i)

   loss.backward()
   with torch.no_grad():
      for param in model.parameters():
         param -= learning_rate*param.grad
         param.grad.zero_()

