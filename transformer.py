from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.loggers import WandbLogger
from torch import Tensor
import torch
import math
import dataset

from torch.optim.optimizer import Optimizer
import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks
DEVICE = torch.device('cuda:0')
import model
   
def initialize_model_parameters(model):
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p) 

class Seq2SeqTransformer(pl.LightningModule):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab: dataset.Vocabulary,
                 tgt_vocab: dataset.Vocabulary,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 batch_size: int = 64,
                 warmup_steps: int = 8000,
                 pad_idx: int = 2
                 ):
        
        super(Seq2SeqTransformer, self).__init__()
        self.save_hyperparameters()
        
        self.transformer = model.NMTmodel(
            src_vocab.stoi, 
            tgt_vocab.stoi, 
            dataset.Vocabulary.PADDING_TOKEN, 
            dataset.Vocabulary.UNKNOWN_WORD_TOKEN, 
            embed_size=emb_size, 
            heads_count=nhead, 
            fnn_hidden_size=dim_feedforward, 
            encoder_layer_count=num_encoder_layers, 
            decoder_layer_count=num_decoder_layers,
            dropoutrate=dropout
        )
        
        initialize_model_parameters(self.transformer)
        
        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.emb_size = emb_size
        self.pad_idx = pad_idx
        
        self.perplexity_accum = 0.0
        self.perplexity_words_count = 0.0
    
    def forward(self, src: Tensor, tgt: Tensor):
        outs = self.transformer(src=src,tgt=tgt)
        return outs
    
    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        def warm_decay(step):
            if step < self.warmup_steps:
                return  step / self.warmup_steps
            return self.warmup_steps ** 0.5 * step ** -0.5
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, warm_decay),
                "interval": "step", #runs per batch rather than per epoch
                "frequency": 1,
            }
        }
    
    def on_before_optimizer_step(self, optimizer: Optimizer, optimizer_idx: int) -> None:
        self.log("lr", optimizer.param_groups[0]['lr'])
    
    def training_step(self, train_batch , batch_idx):
        src, tgt = train_batch # [sentence_length, batch_size]
        src = src.transpose(0,1)# [batch_size, sentence_length]
        tgt = tgt.transpose(0,1)
        
        H = self.transformer(src,tgt, is_train=True)       
        self.log("train_loss",H, batch_size=self.batch_size)
        return H
        
    def on_validation_epoch_end(self) -> None:
        
        self.log(
            "perplexity", math.exp(self.perplexity_accum/self.perplexity_words_count)
        )
        
        self.perplexity_accum = 0.0
        self.perplexity_words_count = 0.0

    def validation_step(self, val_batch, batch_idx):        
        src, tgt = val_batch
        
        src = src.transpose(0,1)
        tgt = tgt.transpose(0,1)
        
        l = sum(len(sentence_tokens)-1 for sentence_tokens in tgt)
        loss = self.transformer(src,tgt)
        
        self.perplexity_accum += loss.item() * l 
        self.perplexity_words_count += l 
        
        self.log("val_loss",loss,batch_size=self.batch_size)        
        return loss
        
    def save(self,fileName):
        torch.save(self.state_dict(), fileName)
    
    def load(self,fileName):
        self.load_state_dict(torch.load(fileName))

def load_test_dataset():
    sourceVocab, targetVocab = dataset.Word_Vocabulary(), dataset.Word_Vocabulary()
    sourceVocab.load(parameters.sourceVocabFile)
    targetVocab.load(parameters.targetVocabFile)
    
    test_dataset = dataset.TranslationDataset(sourceVocab, targetVocab)
    test_dataset.load(parameters.testDatasetFileName)
    
    return test_dataset, len(sourceVocab), len(targetVocab)

    
import parameters
import utils
source_vocab, target_vocab = utils.load_word_vocabulary()

training_dataset = dataset.TranslationDataset(
    source_language_vocab=source_vocab,
    target_language_vocab=target_vocab
)
training_dataset.load(parameters.trainingDatasetFilename)
train_dataloader = dataset.get_data_loader(training_dataset, parameters.BATCH_SIZE, 8)

validation_dataset = dataset.TranslationDataset(
    source_language_vocab=source_vocab,
    target_language_vocab=target_vocab
)
validation_dataset.load(parameters.validationDatasetFilename)
val_dataloader = dataset.get_data_loader(validation_dataset, parameters.BATCH_SIZE, 8, shuffle=False)

nmt = Seq2SeqTransformer(
    parameters.NUM_ENCODER_LAYERS,
    parameters.NUM_DECODER_LAYERS,
    parameters.EMB_SIZE,
    parameters.NHEAD,
    training_dataset.source_language_vocab,
    training_dataset.target_language_vocab,
    parameters.FFN_HID_DIM,
    parameters.DROPOUT_RATE,
    parameters.BATCH_SIZE,
    parameters.WARMUP_STEPS,
    training_dataset.pad_idx
)

early_stop = callbacks.EarlyStopping(
    monitor='perplexity',
    patience=parameters.max_patience,
)

checkpoint_callback = callbacks.ModelCheckpoint(
    monitor='perplexity',
    dirpath='./test',
    filename='seq2seq-epoch{epoch:02d}-perplexity{perplexity:.2f}',
    auto_insert_metric_name=False,
    mode='min'
)   

lr_monitor = callbacks.LearningRateMonitor(logging_interval='step')

wandb_logger = WandbLogger()
trainer = pl.Trainer(
    logger=wandb_logger,
    accelerator='gpu', devices=1,
    accumulate_grad_batches=parameters.GRADIENT_ACCUMULATION_STEP,
    max_epochs=parameters.MAX_TRAINING_EPOCHS,
    val_check_interval=250,
    gradient_clip_val=parameters.clip_grad,
    callbacks=[early_stop, checkpoint_callback, lr_monitor],
    min_epochs=2
)

trainer.fit(nmt, train_dataloader, val_dataloader)

# visualize_custom_lr_adam()


# import sys
# import os
# if len(sys.argv) == 3 and sys.argv[1] == 'translate': 
#     entries = os.listdir('./test/')
#     nmt = nmt.load_from_checkpoint(
#         f"./test/{entries[0]}",
#         num_encoder_layers=  parameters.NUM_ENCODER_LAYERS,
#         num_decoder_layers=parameters.NUM_DECODER_LAYERS,
#         emb_size=parameters.EMB_SIZE,
#         nhead= parameters.NHEAD,
#         src_vocab=training_dataset.source_language_vocab,
#         tgt_vocab= training_dataset.target_language_vocab,
#         dim_feedforward=parameters.FFN_HID_DIM,
#         dropout=parameters.DROPOUT_RATE,
#         batch_size=parameters.BATCH_SIZE,
#         warmup_steps=parameters.WARMUP_STEPS,
#         pad_idx= training_dataset.pad_idx
#     ).to(DEVICE)
    
#     print('_______')
#     test_dataset,_,_ = utils.load_test_dataset(source_vocab=source_vocab, target_vocab=target_vocab)
    
    
#     test_dataloader = dataset.get_data_loader(test_dataset, 1, 8, shuffle=False)

#     nmt.eval()
#     file = open(sys.argv[2],'w')

#     translated_sentences = []
#     for idx, (src,_) in enumerate(test_dataloader):
#         file.write(' '.join(nmt.transformer.translateSentence(src.transpose(0,1).to(DEVICE), beam_size=3, translation_variants_limit=5))+"\n")
#         print(f"Sentence {idx} translated")