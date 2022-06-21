from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

class MyCustomDataset(Dataset):
    def __init__(self, input_, target):
        self.input=input_
        self.target=target
        
    def __getitem__(self, index):
        # stuff
        return (self.input[index,:], self.target[index,:])

    def __len__(self):
        return len(input_)

    

input_n=torch.tensor(stdScaler_input.transform(input_))
target_n=torch.tensor(stdScaler_target.transform(target))

training_data=MyCustomDataset(target_n,input_n)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=False)
