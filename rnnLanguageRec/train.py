from data import *
from network import *
import random
import torch
import time
import math

num_epochs = 100000
criterion = nn.NLLLoss()
learning_rate = .005
current_loss = 0
num_correct = 0
optimizer = torch.optim.SGD(rnn.parameters(), lr = learning_rate)

for i in range(num_epochs):
   category, name, name_tensor, category_tensor = random_example()
   optimizer.zero_grad()

   hidden =rnn.initHidden()
   for l in range(name_tensor.size()[0]):
      output, hidden = rnn(name_tensor[l], hidden)
   loss = criterion(output, category_tensor)
   loss.backward()
   optimizer.step()
   current_loss += loss.item()  
   guess  = get_output(output)
   if category in guess: num_correct += 1
   if(i %5000): 
       print("%d: %f" % (i, num_correct/i),num_correct)
       print(guess, category)
