#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2020/2021
#############################################################################
###
### Невронен машинен превод
###
#############################################################################

import torch
import math
import nltk

class NMTmodel(torch.nn.Module):
    def preparePaddedBatch(self, source, word2ind, pad_token_idx, unk_token_idx):
        device = next(self.parameters()).device
        m = max(len(s) for s in source)
        sents = [[word2ind.get(w,unk_token_idx) for w in s] for s in source]
        sents_padded = [ s+(m-len(s))*[pad_token_idx] for s in sents]
        return torch.tensor(sents_padded, dtype=torch.long, device=device)
    
    def save(self,fileName):
        torch.save(self.state_dict(), fileName)
    
    def load(self,fileName):
        self.load_state_dict(torch.load(fileName))
    
    def __init__(self, source_word2id, target_word2id, pad_token, unk_token, embed_size=512 , heads_count=8, fnn_hidden_size=2048 , encoder_layer_count=6 , decoder_layer_count=6 ,dropoutrate=0.1):
        super(NMTmodel, self).__init__()
        self.source_word2id = source_word2id
        self.target_word2id = target_word2id
        
        self.source_dictionary_pad_token_idx = source_word2id[pad_token]
        self.source_dictionary_unk_token_idx = source_word2id[unk_token]
        
        self.target_dictionary_pad_token_idx = target_word2id[pad_token]
        self.target_dictionary_unk_token_idx = target_word2id[unk_token]
        
        self.encoder = Encoder(len(self.source_word2id),embed_size,heads_count,fnn_hidden_size,encoder_layer_count, dropoutrate=dropoutrate) # What if the source has less words than the target??
        self.decoder = Decoder(len(self.target_word2id), embed_size, heads_count, fnn_hidden_size, decoder_layer_count, dropoutrate=dropoutrate)
        self.projection = torch.nn.Linear(embed_size, len(target_word2id))

        self.tokenizer = nltk.TreebankWordTokenizer()
    
    def forward(self, source, target, is_train=False):
        source_prepared = source #self.preparePaddedBatch(source, self.source_word2id, self.source_dictionary_pad_token_idx, self.source_dictionary_unk_token_idx) #(s,w)
        target_prepared = target #self.preparePaddedBatch(target, self.target_word2id, self.target_dictionary_pad_token_idx, self.target_dictionary_unk_token_idx)

        encoder_pad_mask, decoder_future_mask, encoder_decoder_mask = self._create_masks(source_prepared, target_prepared[:,:-1]) #(s,1,w,w)
        encoder_output = self.encoder(source_prepared, source_mask=encoder_pad_mask) # (s,w,e)
        decoder_output = self.decoder(encoder_output, target_prepared[:,:-1], encoder_decoder_mask=None, decoder_future_mask=decoder_future_mask) #(s,w,e)
        
        predicted_words = self.projection(decoder_output).flatten(0,1) #(s*w, |vocab|)
        expected_words = target_prepared[:,1:].flatten(0,1) #(s*w)

        if is_train:
            return torch.nn.functional.cross_entropy(predicted_words, expected_words, ignore_index=self.target_dictionary_pad_token_idx, label_smoothing=0.1)
        return torch.nn.functional.cross_entropy(predicted_words, expected_words, ignore_index=self.target_dictionary_pad_token_idx)
    
    def translateSentence(self, sentence, start_token = '<SOS>', end_token='<EOS>', beam_size=3,  translation_variants_limit=5, translation_length_limit=1000):
        self.eval()
        sentence_prepared = sentence #self.preparePaddedBatch([sentence], self.source_word2id, self.source_dictionary_pad_token_idx, self.source_dictionary_unk_token_idx) #(1,w) -> (1,w)
        encoder_pad_mask =  torch.bitwise_or((sentence_prepared == self.source_dictionary_pad_token_idx)[:,None,:,None], (sentence_prepared==self.source_dictionary_pad_token_idx)[:,None,None,:])
        encoder_output = self.encoder(sentence_prepared, source_mask = encoder_pad_mask) #(1,w) -> (1,w,e)

        device = next(self.parameters()).device
        
        curr_front = [[self.target_word2id[start_token]]] #(beam_size, 1)
        
        bestTranslation = None
        bestCost = -float('inf')
        
        translation_cost = torch.tensor([[0.0]]*beam_size) #(beam_size,1)
        completed_translation_variations = 0

        target_id2word = {idx:word for word, idx in self.target_word2id.items()}

        while completed_translation_variations < translation_variants_limit and len(curr_front) > 0 and len(curr_front[0]) < translation_length_limit:       
            decoder_input = torch.tensor(curr_front , dtype=torch.long, device=device)
            _, decoder_future_mask, _  = self._create_masks(sentence_prepared, decoder_input)

            resized_encoder_output = encoder_output.repeat(decoder_input.shape[0],1,1)
            decoder_output = self.decoder(resized_encoder_output, decoder_input, encoder_decoder_mask = None, decoder_future_mask= decoder_future_mask)
            
            projections = self.projection(decoder_output[:,-1]) # (beam_size, w , embedding) -> (beam_size, |vocab|)
            predictions = torch.nn.functional.log_softmax(projections, dim=1) # (beam_size, |vocab|) -> (beam_size, |vocab|)
            
            next_word_possiblities = torch.topk(predictions,beam_size, dim=-1) #(beam_size, beam_size)
            
            next_word_idx_possibilities = next_word_possiblities.indices.detach().cpu()
            next_word_cost_possibilities = next_word_possiblities.values.detach().cpu()
            
            total_cost = (translation_cost[:next_word_cost_possibilities.shape[0]] + next_word_cost_possibilities).flatten(0,1) #(beam_size, beam_size) -> (beam_size*beam_size)
            best_indices = torch.topk(total_cost,beam_size).indices # (beam_size)

            next_front = []
            
            cnt = 0
            for i in range(beam_size):
                row = torch.div(best_indices[i],beam_size,rounding_mode='floor')
                column = best_indices[i] % beam_size

                total_translation_cost = total_cost[best_indices[i]]
                next_word_idx = next_word_idx_possibilities[row,column].item()
                
                if next_word_idx == self.target_word2id[end_token]:
                    completed_translation_variations+=1

                    if bestCost < total_translation_cost:
                        bestCost = total_translation_cost
                        bestTranslation = curr_front[row] + [next_word_idx]
                       
                    continue
                
                translation_cost[cnt] = total_translation_cost
                cnt+=1
                next_front.append(curr_front[row] + [next_word_idx])
            

            curr_front = next_front
           
         #(|vocab|)
        
        if bestTranslation is None:
          bestTranslation = curr_front[0]

        human_readable_translation = list(map(lambda word_idx: target_id2word[word_idx] ,bestTranslation[1:]))

        if human_readable_translation[-1] == end_token:
            return human_readable_translation[:-1]
        return human_readable_translation
    
    def _create_masks(self, source, target):
        device = next(self.parameters()).device
        
        encoder_pad_mask = torch.bitwise_or((source == self.source_dictionary_pad_token_idx)[:,None,:,None], (source==self.source_dictionary_pad_token_idx)[:,None,None,:])
        
        encoder_decoder_mask = torch.bitwise_or((target == self.target_dictionary_pad_token_idx)[:,None,:,None], (source==self.source_dictionary_pad_token_idx)[:,None,None,:])
        
        decoder_future_mask = torch.triu(torch.ones((target.size(1), target.size(1)), device=device), diagonal=1)[None,None].bool()
        decoder_pad_mask = torch.bitwise_or((target == self.target_dictionary_pad_token_idx)[:,None,:,None], (target==self.target_dictionary_pad_token_idx)[:,None,None,:])
        decoder_future_mask = torch.bitwise_or(decoder_future_mask, decoder_pad_mask)
        
        return encoder_pad_mask, decoder_future_mask, encoder_decoder_mask
            
    
class Decoder(torch.nn.Module):
    def __init__(self, vocab_size, model_size, head_number, fnn_hidden_size, num_decoder_layers, dropoutrate=0.1):
        super(Decoder, self).__init__()
        
        self.embed_layer = Embedding(vocab_size,model_size)
        self.layers = torch.nn.ModuleList([DecoderLayer(model_size, head_number, fnn_hidden_size, dropoutrate=dropoutrate) for _ in range(num_decoder_layers)])
        self.norm = torch.nn.LayerNorm(model_size)
    
    def forward(self, encoder_output, decoder_input, encoder_decoder_mask=None,decoder_future_mask=None):
        decoder_output = self.embed_layer(decoder_input)
        for layer in self.layers:
            decoder_output = layer(encoder_output,decoder_output,encoder_decoder_mask=encoder_decoder_mask,decoder_future_mask=decoder_future_mask)
            
        return self.norm(decoder_output)


class Encoder(torch.nn.Module):
    def __init__(self, vocab_size, model_size, head_number, ffn_hidden_size, num_encoder_layers, dropoutrate=0.1):
        super(Encoder, self).__init__()
        
        self.embed_layer = Embedding(vocab_size,model_size)
        self.layers = torch.nn.ModuleList([EncoderLayer(model_size, head_number, ffn_hidden_size, dropoutrate=dropoutrate) for _ in range(num_encoder_layers)])
        self.norm = torch.nn.LayerNorm(model_size)
    
    def forward(self, encoder_input, source_mask=None):
        encoder_output = self.embed_layer(encoder_input)
        for layer in self.layers:
            encoder_output = layer(encoder_output,mask=source_mask) 
                      
        return self.norm(encoder_output)
    

class DecoderLayer(torch.nn.Module):
    def __init__(self, model_size, head_number, ffn_hidden_size,dropoutrate=0.1):
        super(DecoderLayer, self).__init__()
        
        self.multi_self_attention_layer = MultiheadedAttentionNode(model_size,head_number)
        self.self_attention_normalization_layer = torch.nn.LayerNorm(model_size)
        self.dropout_self_attention = torch.nn.Dropout(dropoutrate)
        
        self.encoder_decoder_attention_layer = MultiheadedAttentionNode(model_size,head_number)
        self.attention_normalization_layer = torch.nn.LayerNorm(model_size)
        self.dropout_attention = torch.nn.Dropout(dropoutrate)
        
        self.ffn = PointwiseFeedForwardNetwork(model_size, ffn_hidden_size)
        self.ffn_normalization_layer = torch.nn.LayerNorm(model_size)
        self.dropout_ffn = torch.nn.Dropout(dropoutrate)
    
    def forward(self, encoder_output, decoder_input, encoder_decoder_mask=None, decoder_future_mask=None):
        
        normalized_decoder_inpit = self.self_attention_normalization_layer(decoder_input)
        self_att = self.multi_self_attention_layer(normalized_decoder_inpit, normalized_decoder_inpit, normalized_decoder_inpit, mask=decoder_future_mask)
        queries = decoder_input + self.dropout_self_attention(self_att)
        
        normalized_queries = self.attention_normalization_layer(queries)
        att = self.encoder_decoder_attention_layer(normalized_queries, encoder_output, encoder_output, mask=encoder_decoder_mask)
        att = queries + self.dropout_attention(att)
        
        ffn_input = self.ffn_normalization_layer(att)
        fnn_output = self.ffn(ffn_input)
        decoder_output = att + self.dropout_ffn(fnn_output)
        return decoder_output   
    
class EncoderLayer(torch.nn.Module):
    def __init__(self,model_size,head_number, ffn_hidden_size, dropoutrate=0.1):
        super(EncoderLayer, self).__init__()
        
        self.attention = MultiheadedAttentionNode(model_size, head_number)
        self.attention_normalization = torch.nn.LayerNorm(model_size)
        self.dropout_attention_output = torch.nn.Dropout(dropoutrate)
        
        self.ffn = PointwiseFeedForwardNetwork(model_size, ffn_hidden_size)
        self.ffn_normalization = torch.nn.LayerNorm(model_size)
        self.dropout_ffn_output = torch.nn.Dropout(dropoutrate)
    
    def forward(self, source, mask=None):
        normalized_source = self.attention_normalization(source)
        attention_output = self.attention(normalized_source,normalized_source,normalized_source, mask=mask)
        attention = source + self.dropout_attention_output(attention_output)
        
        normalized_attention = self.ffn_normalization(attention)
        ffn_output = self.ffn(normalized_attention)
        normalized_output = attention + self.dropout_ffn_output(ffn_output) #maybe it sohuld be another node
        
        return normalized_output


class MultiheadedAttentionNode(torch.nn.Module):
    def __init__(self, model_size, heads_count):
        super(MultiheadedAttentionNode,self).__init__()
        self.hidden_size = model_size // heads_count
        self.heads_count = heads_count
        
        self.key_projection = torch.nn.Linear(model_size,model_size)
        self.value_projection = torch.nn.Linear(model_size,model_size)
        self.query_projection = torch.nn.Linear(model_size,model_size)
        
        self.projection = torch.nn.Linear(self.hidden_size * heads_count,model_size)
        
    def forward(self, queries, keys, values, mask=None): #(s, w, model_size)
        batch_size, output_seq_len = queries.size(0), queries.size(1)
    
        queries = self.query_projection(queries).view(batch_size,queries.size(1),-1, self.hidden_size).transpose(2,1) #(batch_size, sequence_len, embedding) -> (batch_size, heads_count, sequence_len, d_model // heads_count)
        keys = self.key_projection(keys).view(batch_size,keys.size(1),-1, self.hidden_size).transpose(2,1) 
        values = self.value_projection(values).view(batch_size,values.size(1),-1, self.hidden_size).transpose(2,1) 
        
        attention_scores = torch.matmul(queries, keys.transpose(-1,-2)) / math.sqrt(self.hidden_size) #(batch_size, heads_count, sequence_len, hidden_count) * (batch_size, heads_count, hidden_count, sequence_len) =  (batch_size, heads_count, sequence_len, sequence_len)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask, -float(1e15))
        
        att_vec = torch.matmul(torch.nn.functional.softmax(attention_scores,dim=-1), values) # (s, h, w, w) * (s,h,w,k) -> (s,h,w,k)
    
        return self.projection(att_vec.transpose(2,1).contiguous().view(batch_size, output_seq_len, -1)) # check if this implementation is correct

class PointwiseFeedForwardNetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_size, dropoutrate=0.1):
        super(PointwiseFeedForwardNetwork, self).__init__()
        self.hidden_layer = torch.nn.Linear(input_size, hidden_size)
        self.projection = torch.nn.Linear(hidden_size, input_size) 
        self.dropout = torch.nn.Dropout(dropoutrate)  
    
    def forward(self, source):
        hidden_vectors = self.dropout(torch.relu(self.hidden_layer(source)))
        return self.projection(hidden_vectors)

class Embedding(torch.nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(Embedding, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, emb_size)
        self.positional_embed = PositionalEncoding(emb_size)#,requires_grad = False)
        self.emb_size = emb_size

    def forward(self, tokens):
        token_embed = self.embedding(tokens.long()) * math.sqrt(self.emb_size)
        return token_embed + self.positional_embed(token_embed.size(1))


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len = 1000): # max_len is a limit over the length of words of a sentence (aka record)
        super(PositionalEncoding, self).__init__()
        
        position = torch.arange(max_len).unsqueeze(1) # (1,max_len) -> (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, seq_len):
        return self.pe[0,:seq_len]

