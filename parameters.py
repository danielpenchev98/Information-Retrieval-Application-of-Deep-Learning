import torch

sourceFileName = 'en_bg_data/train.en'
targetFileName = 'en_bg_data/train.bg'
sourceDevFileName = 'en_bg_data/dev.en'
targetDevFileName = 'en_bg_data/dev.bg'
sourceTestFileName = 'en_bg_data/test.en'
targetTestFileName = 'en_bg_data/test.bg'

corpusDataFileName = 'corpusData'
wordsDataFileName = 'wordsData'
modelFileName = 'NMTmodel'

trainingDatasetFilename ='trainingDataset'
validationDatasetFilename = 'validationDataset'
testDatasetFileName = 'testDataset'
sourceVocabFile = "sourceVocab"
targetVocabFile = "targetVocab"

device = torch.device("cuda:0")

EMB_SIZE = 192
NHEAD = 8
FFN_HID_DIM = 1024
BATCH_SIZE = 96
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

WARMUP_STEPS = 2000
MAX_TRAINING_EPOCHS = 160

MAX_SENTENCE_TOKENS=10000
MIN_SENTENCE_TOKENS=0

GRADIENT_ACCUMULATION_STEP=2

DROPOUT_RATE = 0.1

uniform_init = 0.1
learning_rate = 0.001
clip_grad = 1.0

FREQ_THRESHOLD=2

max_patience = 10
max_trials = 10
