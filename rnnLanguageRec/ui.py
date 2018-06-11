from network import*
from data import*
rnn = torch.load('name-classification.pt')		
def predict(line):
   line_tensor = line_to_tensor(line)
   hidden = rnn.initHidden()
   for i in range(line_tensor.size()[0]):
      output, hidden = rnn(line_tensor[i], hidden)
   print(get_output(output))

user = " "
while len(user) > 0:
   user = input("Enter a name: ")
   if len(user) == 0: break
   predict(user)
 
