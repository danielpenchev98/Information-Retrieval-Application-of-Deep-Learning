import unittest

import torch

from dataset import Train_Dataset

class TrainDatasetTest(unittest.TestCase):
    def test_get_batch(self):
        dataset = Train_Dataset()
        dataset.numericalized_source_sentences=[torch.tensor(i) for i in range(0,20)]
        dataset.numericalized_target_sentences=[torch.tensor(i) for i in range(100,120)]
        
        dataset.order = torch.arange(0,20)
        batch_src, batch_tgt = dataset.get_batch(0,10)
        print(batch_src)
        print(batch_tgt)
        
        dataset.shuffle()
        
        batch_src, batch_tgt = dataset.get_batch(0,10)
        print(batch_src)
        print(batch_tgt)
        

if __name__ == '__main__':
    unittest.main()