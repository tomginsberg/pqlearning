from torch.utils.data import Dataset

class Subsetable(Dataset):
    def __init__(self, dataset):
        self.indices = list(range(len(dataset)))
        self.dataset = dataset



    def refine_dataset(self, indices):
        """
        Refines the dataset by keeping only the indices specified in the argument.
        :param indices: list of indices to keep
        """
        self.indices = [self.indices[i] for i in indices]

    def __getitem__(self, item):
        return self.dataset[self.indices[item]]


    def __len__(self):
        return len(self.indices)
