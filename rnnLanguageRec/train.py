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
def evaluate():
    num_correct = 0
    batch_size = 10000
    for i in range(batch_size):
        category, name, name_tensor, category_tensor = random_example()
        hidden = rnn.initHidden()
        for l in range(name_tensor.size()[0]):
            output, hidden = rnn(name_tensor[l], hidden)
        guess = get_output(output)
        if category in guess: num_correct += 1
        if(i %500==0 and i>0): 
           print("%d: %f" % (i, num_correct/i),num_correct)
           print(guess, category)

def train():
   for i in range(num_epochs):
      category, name, name_tensor, category_tensor = random_example()
      optimizer.zero_grad()

      hidden =rnn.initHidden()
      for l in range(name_tensor.size()[0]):
         output, hidden = rnn(name_tensor[l], hidden)
      loss = criterion(output, category_tensor)
      loss.backward()
      optimizer.step()
      global current_loss
      current_loss += loss.item()  
      if i % 5000==0: 
          print("%d: %f" % (i,loss.item() ))

   torch.save(rnn, "name-classification.pt")
def run():
    train()
    evaluate()

if __name__ == '__main__':
    run()
