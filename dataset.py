from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import nltk
import torch
import pickle
import os

from typing import List
from parameters import device
import sentencepiece as spm

class Vocabulary:
    PADDING_TOKEN = '<PAD>'
    START_OF_SENTENCE_TOKEN = '<SOS>'
    END_OF_SENTENCE_TOKEN = '<EOS>'
    UNKNOWN_WORD_TOKEN = '<UNK>'
    
    def __len__(self):
        raise NotImplementedError
    
    def numericalize(self, text_sentences: List[str], is_src=False) -> List[torch.Tensor]:
        raise NotImplementedError
    
    def denumericalize(self, text: torch.Tensor) -> str:
        raise NotImplementedError
    
    def build_vocabulary(self, text_sentences):
        raise NotImplementedError
    
    def to_idx(self, token: str) -> int:
        raise NotImplementedError
    
    def to_str(self, idx: int) -> str:
        raise NotImplementedError
    
    def save(self, filePath):
        raise NotImplementedError

    def load(self, filePath):
        raise NotImplementedError


class BPE_Vocabulary(Vocabulary):
    
    def __init__(self, model_filename_prefix="", vocab_size=0, is_src=False):
        super(BPE_Vocabulary, self).__init__()
        self.sp = None
        self.is_src=is_src
        self.temp_storage_file = f"{model_filename_prefix}.temp_storage.txt"
        self.vocab_size = vocab_size
        self.model_filename_prefix = model_filename_prefix
    
    def __len__(self):
        return self.sp.GetPieceSize() if self.sp else self.vocab_size
        
    def tokenize(self, text: str):
        return self.sp.Encode(text, out_type=str)
        
    def numericalize(self, text_sentences: List[str]) -> List[torch.Tensor]:
        numericalized_text = self.sp.Encode(text_sentences, out_type=int, add_bos=True, add_eos=True)
        return list(map(lambda x: torch.as_tensor(x, dtype=torch.long), numericalized_text))
    
    def denumericalize(self, text: torch.Tensor) -> str:
        text = text.tolist()
        print(text)
        return self.sp.Decode(text, out_type=str)
    
    def to_idx(self, token: str) -> int:
        return self.sp.PieceToId(token)
    
    def to_str(self, idx: int) -> str:
        return self.sp.IdToPiece(idx)
    
    def build_vocabulary(self, text_sentences):
        with open(self.temp_storage_file, 'w') as file:
            for sent in text_sentences:
                file.write(sent +"\n")
        
        pad_id, bos_id, eos_id, unk_id = 0, 1, 2, 3
        if self.is_src:
            pad_id, bos_id, eos_id, unk_id = 0, -1, -1, 1
        
        spm.SentencePieceTrainer.Train(
            input=self.temp_storage_file,
            model_prefix=self.model_filename_prefix,
            vocab_size=self.vocab_size,
            pad_id=pad_id,
            bos_id=bos_id,
            eos_id=eos_id,
            unk_id=unk_id,
            unk_piece=BPE_Vocabulary.UNKNOWN_WORD_TOKEN,
            bos_piece=BPE_Vocabulary.START_OF_SENTENCE_TOKEN,
            eos_piece=BPE_Vocabulary.END_OF_SENTENCE_TOKEN,
            pad_piece=BPE_Vocabulary.PADDING_TOKEN,
            model_type="bpe",
            unk_surface=BPE_Vocabulary.UNKNOWN_WORD_TOKEN,
        )
        
        self.sp = spm.SentencePieceProcessor(model_file=f"{self.model_filename_prefix}.model")
        os.remove(self.temp_storage_file)
        
    def save(self, filePath):
        pickle.dump((self.model_filename_prefix, self.vocab_size, self.is_src), open(filePath, 'wb'))
    
    def load(self, filePath):
        (self.model_filename_prefix,self.vocab_size, self.is_src) = pickle.load(open(filePath, 'rb'))
        self.temp_storage_file = f"{self.model_filename_prefix}.temp_storage.txt"
        self.sp = spm.SentencePieceProcessor(model_file=f"{self.model_filename_prefix}.model")  
           

class Word_Vocabulary(Vocabulary):
    
    def __init__(self, freq_threshold=2, max_size=0, is_src=False):
        '''
        freq_threshold : the minimum times a word must occur in corpus to be added to the vocab
        max_size : max source voca size. Eg. if set to 10,000 we pick the top 10,000 most frequent words and discard others
        '''
        
        super(Word_Vocabulary, self).__init__()
        
        # initialize the index to token dict
        ## <PAD> -> padding, used for padding shorter sentences in a batch to match the length of longest sentence in the batch
        ## <SOS> -> start token, added in front of each sentence to signify the start of sentence
        ## <EOS> -> end of sentence token, added to the end of each sentence to signify the end of sentence
        ## <UNK> -> words, which are not found in the vocab are replaced by this token
        
        self.itos = {
            0: Word_Vocabulary.PADDING_TOKEN, 
            1: Word_Vocabulary.START_OF_SENTENCE_TOKEN,
            2: Word_Vocabulary.END_OF_SENTENCE_TOKEN,
            3: Word_Vocabulary.UNKNOWN_WORD_TOKEN
        }
        
        #initiate the token to index dict
        self.stoi = {k:j for j,k in self.itos.items()}
        
        self.freq_threshold = freq_threshold
        self.max_size = max_size
        
        self.word_tokenizer = nltk.TreebankWordTokenizer()
        self.word_detokenizer = nltk.TreebankWordDetokenizer()
        self.is_src = is_src
        
    def __len__(self):
        return len(self.itos)
    
    def tokenize(self, sentence):
        return self.word_tokenizer.tokenize(sentence.replace("<UNK>", "UNK"))
    
    def detokenize(self, tokens):
        return self.word_detokenizer.detokenize(tokens)
    
    def to_idx(self, token: str) -> int:
        return self.stoi.get(token, self.stoi[Word_Vocabulary.UNKNOWN_WORD_TOKEN])
    
    def to_str(self, idx: int) -> str:
        return self.itos.get(idx, Word_Vocabulary.UNKNOWN_WORD_TOKEN)
    
    
    def build_vocabulary(self, text_sentences):
        frequencies = {}
        
        for sentence in text_sentences:
            for word in self.tokenize(sentence):
                if word not in frequencies.keys():
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1
        
        frequencies = {k:v for k,v in frequencies.items() if v > self.freq_threshold}
        
        idx = len(self.stoi) # idx from which we want out dict to start
        frequencies = dict(sorted(frequencies.items(), key=lambda x: -x[1])[:self.max_size-idx])
        
        for word in frequencies.keys():
            self.stoi[word] = idx
            self.itos[idx] = word
            idx+=1
    
    def numericalize(self, text_sentences: List[str]) -> List[torch.Tensor]:    
        tokenized_text_sentences = [self.tokenize(sentence) for sentence in text_sentences]
        
        if not self.is_src:
            tokenized_text_sentences = map(lambda s: [Vocabulary.START_OF_SENTENCE_TOKEN] + s + [Vocabulary.END_OF_SENTENCE_TOKEN], tokenized_text_sentences)
        
        numericalized_text = []
        for tokenized_sentence in tokenized_text_sentences:
            numericalized_sentence = []
            for token in tokenized_sentence:
                if token in self.stoi.keys():
                    numericalized_sentence.append(self.stoi[token])
                else:
                    numericalized_sentence.append(self.stoi[Vocabulary.UNKNOWN_WORD_TOKEN])
            numericalized_text.append(torch.as_tensor(numericalized_sentence, dtype=torch.long))
        
        return numericalized_text
    
    def denumericalize(self, text: torch.Tensor) -> str:
        text = text.tolist()
        denumericalized_text = []
        for idx in text:
            if idx in [self.stoi[Vocabulary.START_OF_SENTENCE_TOKEN], self.stoi[Vocabulary.PADDING_TOKEN], self.stoi[Vocabulary.END_OF_SENTENCE_TOKEN]]:
                continue
            denumericalized_text.append(self.itos[idx])
        return self.detokenize(denumericalized_text)
    
    def save(self, filePath):
        pickle.dump((self.itos, self.stoi, self.freq_threshold, self.max_size, self.is_src), open(filePath, 'wb'))

    def load(self, filePath):
        (self.itos,self.stoi,self.freq_threshold,self.max_size, self.is_src) = pickle.load(open(filePath, 'rb'))

class TranslationDataset(Dataset):
    
    def __init__(self, source_language_vocab, target_language_vocab, source_text_sentences=[], target_text_sentences=[], min_sentence_tokens=4, max_sentence_tokens=150):
        self.source_language_vocab = source_language_vocab
        self.target_language_vocab = target_language_vocab
        
        self.pad_idx = self.source_language_vocab.to_idx(Vocabulary.PADDING_TOKEN)
                
        self.numericalized_source_sentences = self.source_language_vocab.numericalize(source_text_sentences)
        self.numericalized_target_sentences = self.target_language_vocab.numericalize(target_text_sentences)  
        
        #self._filter_sentences(min_sentence_tokens, max_sentence_tokens)
        
        self.order = torch.arange(len(self.numericalized_source_sentences), dtype=torch.int32)
        
    def __len__(self):
        return len(self.numericalized_source_sentences)
    
    '''
    __getitem__ runs on 1 example at a time. Here, we get an example at index and return its numericalize source and
    target values using the vocabulary objects we created in __init__
    '''
    
    def __getitem__(self, index):
        return self.numericalized_source_sentences[index], self.numericalized_target_sentences[index]


    def shuffle(self):
        self.order = torch.randperm(len(self.numericalized_source_sentences), dtype=torch.int32)

    def get_batch(self, start_index, batch_size):
        end_index = min(self.__len__(), start_index + batch_size)
      
        batch_source_sentences = [self.numericalized_source_sentences[i] for i in self.order[start_index:end_index]]
        batch_target_sentences = [self.numericalized_target_sentences[i] for i in self.order[start_index:end_index]]
        
        source_sentences = torch.nn.utils.rnn.pad_sequence(batch_source_sentences, batch_first=False, padding_value = self.pad_idx)
        target_sentences = torch.nn.utils.rnn.pad_sequence(batch_target_sentences, batch_first=False, padding_value = self.pad_idx)
        
        return source_sentences, target_sentences
    
    def save(self, datasetFilePath):
        pickle.dump((self.numericalized_source_sentences, self.numericalized_target_sentences ,self.order, self.pad_idx), open(datasetFilePath, 'wb'))
        
    def load(self, datasetFilePath):
        (self.numericalized_source_sentences, self.numericalized_target_sentences, self.order, self.pad_idx) = pickle.load(open(datasetFilePath, 'rb'))
        self.shuffle()

    def _filter_sentences(self, min_tokens=4, max_tokens=150):
        to_swap_idx = len(self.numericalized_source_sentences)-1
        
        for i in range(len(self.numericalized_source_sentences)-1, -1, -1):
            curr_src_sentence, curr_tgt_sentence = self.numericalized_source_sentences[i], self.numericalized_target_sentences[i]
            
            #taking into consideration the SOS and EOS tokens
            if len(curr_src_sentence) - 2 > max_tokens:
                #last token is always the end token
                self.numericalized_source_sentences[i] = self.numericalized_source_sentences[i][:max_tokens+1]
                self.numericalized_target_sentences[i] = self.numericalized_target_sentences[i][:max_tokens+1]
                continue
            
            if min_tokens <= len(curr_src_sentence) - 2:
                continue
            
            self.numericalized_source_sentences[i] = self.numericalized_source_sentences[to_swap_idx]
            self.numericalized_source_sentences[to_swap_idx] =  curr_src_sentence
            
            self.numericalized_target_sentences[i] = self.numericalized_target_sentences[to_swap_idx]
            self.numericalized_target_sentences[to_swap_idx] = curr_tgt_sentence
            
            to_swap_idx -= 1
        
        #to_swap_index always points to the last sentence that is over the min_tokens length from left to right
        self.numericalized_source_sentences = self.numericalized_source_sentences[:to_swap_idx+1]
        self.numericalized_target_sentences = self.numericalized_target_sentences[:to_swap_idx+1]

'''
class to add padding to the batches
collat_fn in dataloader is used for post processing on a single batch. Like __getitem__ in dataset class
is used on single example
'''

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
    
    ## First the object is created using MyCollate(pad_idx) in data loader
    ## Then if obj(batch) is called -> __call__ runs by default
    def __call__(self, batch):
        # get all source indexed sentences of the batch
        source = [item[0] for item in batch]
        #pad them using pad_sequence method from pytorch
        source = torch.nn.utils.rnn.pad_sequence(source, batch_first=False, padding_value = self.pad_idx)
        
        #get all target indexed sentences of the batch
        target = [item[1] for item in batch]
        #pad them using pad_sequence method from pytorch
        target = torch.nn.utils.rnn.pad_sequence(target, batch_first=False, padding_value=self.pad_idx)
        
        return source, target

def get_data_loader(dataset, batch_size, num_workers=0, shuffle=True, pin_memory=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=dataset.pad_idx)
    )