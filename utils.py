import sys
import torch
import dataset
from nltk.translate.bleu_score import corpus_bleu
from parameters import *
DEVICE = torch.device('cuda:0')

def prepareBpeDataset():
    train_source_text_sentences = open(sourceFileName).read().splitlines()
    train_target_text_sentences = open(targetFileName).read().splitlines()
    
    #train_source_text_sentences, train_target_text_sentences = find_duplicate_translations(train_source_text_sentences,train_target_text_sentences)
    
    source_bpe_vocab = dataset.BPE_Vocabulary(model_filename_prefix=sourceVocabFile, vocab_size=12000)
    source_bpe_vocab.build_vocabulary(train_source_text_sentences)
    source_bpe_vocab.save(sourceVocabFile)
    
    target_bpe_vocab = dataset.BPE_Vocabulary(model_filename_prefix=targetVocabFile, vocab_size=22000)
    target_bpe_vocab.build_vocabulary(train_target_text_sentences)
    target_bpe_vocab.save(targetVocabFile)
    
    training_dataset = dataset.TranslationDataset(
        source_language_vocab=source_bpe_vocab,
        target_language_vocab=target_bpe_vocab,
        source_text_sentences=train_source_text_sentences,
        target_text_sentences=train_target_text_sentences,
        max_sentence_tokens=MAX_SENTENCE_TOKENS,
        min_sentence_tokens=MIN_SENTENCE_TOKENS,
    )
    training_dataset.save(trainingDatasetFilename)

    val_source_text = open(sourceDevFileName).read().splitlines()
    val_target_text = open(targetDevFileName).read().splitlines()
    
    validation_dataset = dataset.TranslationDataset(
        source_language_vocab=source_bpe_vocab,
        target_language_vocab=target_bpe_vocab,
        source_text_sentences=val_source_text,
        target_text_sentences=val_target_text,
        max_sentence_tokens=MAX_SENTENCE_TOKENS,
        min_sentence_tokens=MIN_SENTENCE_TOKENS,
    )
    validation_dataset.save(validationDatasetFilename)

    
    test_source_text = open(sourceTestFileName).read().splitlines()
    test_target_text = open(targetTestFileName).read().splitlines()
    
    test_dataset = dataset.TranslationDataset(
        source_language_vocab=source_bpe_vocab,
        target_language_vocab=target_bpe_vocab,
        source_text_sentences=test_source_text,
        target_text_sentences=test_target_text,
        max_sentence_tokens=MAX_SENTENCE_TOKENS,
        min_sentence_tokens=MIN_SENTENCE_TOKENS,
    )
    test_dataset.save(testDatasetFileName)

def prepareDatasets():
    train_source_text_sentences = open(sourceFileName).read().splitlines()
    train_target_text_sentences = open(targetFileName).read().splitlines()
    
    source_word_vocab = dataset.Word_Vocabulary()
    source_word_vocab.build_vocabulary(train_source_text_sentences)
    source_word_vocab.save(sourceVocabFile)
    
    target_word_vocab = dataset.Word_Vocabulary()
    target_word_vocab.build_vocabulary(train_target_text_sentences)
    target_word_vocab.save(targetVocabFile)
    
    #train_source_text_sentences, train_target_text_sentences = find_duplicate_translations(train_source_text_sentences,train_target_text_sentences)
    
    training_dataset = dataset.TranslationDataset(
        source_language_vocab=source_word_vocab,
        target_language_vocab=target_word_vocab,
        source_text_sentences=train_source_text_sentences,
        target_text_sentences=train_target_text_sentences,
        max_sentence_tokens=MAX_SENTENCE_TOKENS,
        min_sentence_tokens=MIN_SENTENCE_TOKENS,
    )
    training_dataset.save(trainingDatasetFilename)

    val_source_text = open(sourceDevFileName).read().splitlines()
    val_target_text = open(targetDevFileName).read().splitlines()
    
    validation_dataset = dataset.TranslationDataset(
        source_language_vocab=source_word_vocab,
        target_language_vocab=target_word_vocab,
        source_text_sentences=val_source_text,
        target_text_sentences=val_target_text,
        max_sentence_tokens=MAX_SENTENCE_TOKENS,
        min_sentence_tokens=MIN_SENTENCE_TOKENS,
    )
    validation_dataset.save(validationDatasetFilename)

    
    test_source_text = open(sourceTestFileName).read().splitlines()
    test_target_text = open(targetTestFileName).read().splitlines()
    
    test_dataset = dataset.TranslationDataset(
        source_language_vocab=source_word_vocab,
        target_language_vocab=target_word_vocab,
        source_text_sentences=test_source_text,
        target_text_sentences=test_target_text,
        max_sentence_tokens=MAX_SENTENCE_TOKENS,
        min_sentence_tokens=MIN_SENTENCE_TOKENS,
    )
    test_dataset.save(testDatasetFileName)
    
def find_duplicate_translations(source_sentences, target_sentences):
    
    deduplicated_source_sentences = []
    deduplicated_target_sentences = []
    cache = {}
    for (src_sentence, tgt_sentence) in zip(source_sentences, target_sentences):
        
        if src_sentence not in cache:
            cache[src_sentence] = {tgt_sentence}
            deduplicated_source_sentences.append(src_sentence)
            deduplicated_target_sentences.append(tgt_sentence)
            continue
        
        cached_tgt_sentences = cache[src_sentence]
        if tgt_sentence in cached_tgt_sentences:
            continue
        
        cached_tgt_sentences.add(tgt_sentence)
        deduplicated_source_sentences.append(src_sentence)
        deduplicated_target_sentences.append(tgt_sentence)
    
    return deduplicated_source_sentences, deduplicated_target_sentences

def load_bpe_vocabulary():
    source_bpe_vocab = dataset.BPE_Vocabulary()
    source_bpe_vocab.load(sourceVocabFile)
    
    target_bpe_vocab = dataset.BPE_Vocabulary()
    target_bpe_vocab.load(targetVocabFile)
    
    return source_bpe_vocab, target_bpe_vocab

def load_word_vocabulary():
    source_bpe_vocab = dataset.Word_Vocabulary()
    source_bpe_vocab.load(sourceVocabFile)
    
    target_bpe_vocab = dataset.Word_Vocabulary()
    target_bpe_vocab.load(targetVocabFile)
    
    return source_bpe_vocab, target_bpe_vocab

def load_train_val_datasets(source_vocab: dataset.Vocabulary, target_vocab: dataset.Vocabulary):
    training_dataset = dataset.TranslationDataset(
        source_language_vocab=source_vocab,
        target_language_vocab=target_vocab,
    )
    training_dataset.load(trainingDatasetFilename)
    
    validation_dataset = dataset.TranslationDataset(
        source_language_vocab=source_vocab,
        target_language_vocab=target_vocab,
    )
    validation_dataset.load(validationDatasetFilename)
    
    return training_dataset, validation_dataset

def load_test_dataset(source_vocab: dataset.Vocabulary, target_vocab: dataset.Vocabulary):
    test_dataset = dataset.TranslationDataset(
        source_language_vocab=source_vocab,
        target_language_vocab=target_vocab,
    )
    test_dataset.load(testDatasetFileName)
    
    return test_dataset, len(source_vocab), len(target_vocab)
         
if len(sys.argv)>1 and sys.argv[1] == 'prepare':
    if len(sys.argv) == 3 and sys.argv[2] == "bpe":
        prepareBpeDataset()
    else:    
        prepareDatasets()

if len(sys.argv)>3 and sys.argv[1] == 'bleu':
    fileName = sys.argv[2]
    
    _, target_vocab = load_word_vocabulary()
    
    ref = [[target_vocab.tokenize(line)] for line in open(sys.argv[2]) ]
    hyp = [target_vocab.tokenize(line) for line in open(sys.argv[3]) ]
    
    bleu_score = corpus_bleu(ref, hyp)
    print('Corpus BLEU: ', (bleu_score * 100))