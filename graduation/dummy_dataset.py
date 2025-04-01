from paddle.io import Dataset
import numpy as np
import paddle

class DummyDataset(Dataset):
    def __init__(self, vocab_size, seq_length):
        super(DummyDataset, self).__init__()
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.generate_dummy_data()
    
    def generate_dummy_data(self):
        self.dataset_size = 512 * 20  # 暂且临时设置为512 * 20
        self.input_list = []
        self.label_list = []
        
        for _ in range(self.dataset_size):
            single_sentence_length = np.random.randint(1, self.seq_length + 1) # [1, seq_length + 1)
            input = np.random.randint(0, self.vocab_size, size=(self.seq_length,), dtype=np.int64) # [note] 类型为np.int64,即8字节
            input[single_sentence_length:] = 0
            label = np.zeros_like(input)
            label[:-1] = input[1:self.seq_length]
            self.input_list.append(input)
            self.label_list.append(label)
    
    def __getitem__(self, idx):
        if idx >= self.dataset_size:
            raise IndexError("Index out of range")
        input, lable = paddle.to_tensor(self.input_list[idx]), paddle.to_tensor(self.label_list[idx])
        return input, lable
    
    def __len__(self):
        return self.dataset_size
            