import torch.nn as nn
import torch
import data
from torch.autograd import Variable
class RNN(nn.Module):
   def __init__(self, input_size, h_size, output_size):
      super(RNN, self).__init__()
      
      self.h_size = h_size
      self.i2h = nn.Linear(input_size + h_size, h_size)
      self.i2o = nn.Linear(input_size + h_size, output_size)
      self.softmax = nn.LogSoftmax(dim = 1)

   def forward(self, input, hidden):
      combined = torch.cat((input, hidden), 1)
      hidden = self.i2h(combined)
      output = self.i2o(combined)
      output = self.softmax(output)
      return output, hidden

   def initHidden(self):
      return Variable(torch.zeros(1, self.h_size))

def get_output(output):
   _,top_i = output.topk(3)
   cat_index =[ top_i[0][i] for i in range(top_i[0].size()[0])]
   top_cats = []
   for index in cat_index:
     top_cats.append(data.all_categories[index])
   return top_cats
n_hidden = 128
rnn = RNN(data.n_letters, n_hidden, data.n_categories)

