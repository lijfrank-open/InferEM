import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
from code.common_layer import EncoderLayer, EncoderLayer_cr ,DecoderLayer, MultiHeadAttention, Conv, PositionwiseFeedForward, LayerNorm , _gen_bias_mask ,_gen_timing_signal, share_embedding, LabelSmoothing, NoamOpt, _get_attn_subsequent_mask
import random
import os
import pprint
from tqdm import tqdm
pp = pprint.PrettyPrinter(indent=1)
import os
import time
from copy import deepcopy
from sklearn.metrics import accuracy_score
import pdb


torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)


class Encoder(nn.Module):
    """
    A Transformer Encoder module. 
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self, args, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=1000, input_dropout=0.0, layer_dropout=0.0, 
                 attention_dropout=0.0, relu_dropout=0.0, use_mask=False, universal=False, concept=False):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder  2
            num_heads: Number of attention heads   2
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head   40
            total_value_depth: Size of last dimension of values. Must be divisible by num_head  40
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN  50
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
            use_mask: Set to True to turn on future value masking
        """

        super(Encoder, self).__init__()
        self.args = args
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)
        
        if(self.universal):  

            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        params =(hidden_size, 
                 total_key_depth or hidden_size,
                 total_value_depth or hidden_size,
                 filter_size, 
                 num_heads, 
                 _gen_bias_mask(max_length) if use_mask else None,
                 layer_dropout, 
                 attention_dropout, 
                 relu_dropout)
        
        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        if(self.universal):
            self.enc = EncoderLayer(*params)
        else:
            self.enc = nn.ModuleList([EncoderLayer(*params) for _ in range(num_layers)])
        
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs, mask):

        x = self.input_dropout(inputs)
        
        x = self.embedding_proj(x)
        
        if(self.universal):
            if(self.args.act):  
                x, (self.remainders, self.n_updates) = self.act_fn(x, inputs, self.enc, self.timing_signal, self.position_signal, self.num_layers)
                y = self.layer_norm(x)
            else:
                for l in range(self.num_layers):
                    x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
                    x += self.position_signal[:, l, :].unsqueeze(1).repeat(1,inputs.shape[1],1).type_as(inputs.data)
                    x = self.enc(x, mask=mask)
                y = self.layer_norm(x)
        else:
            
            x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
            
            for i in range(self.num_layers):
                x = self.enc[i](x, mask)
        
            y = self.layer_norm(x)
        return y

class Encoder_cr(nn.Module):
    """
    A Transformer Encoder module. 
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self, args, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=1000, input_dropout=0.0, layer_dropout=0.0, 
                 attention_dropout=0.0, relu_dropout=0.0, use_mask=False, universal=False, concept=False):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder  2
            num_heads: Number of attention heads   2
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head   40
            total_value_depth: Size of last dimension of values. Must be divisible by num_head  40
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN  50
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
            use_mask: Set to True to turn on future value masking
        """

        super(Encoder_cr, self).__init__()
        self.args = args
        self.universal = universal
        self.num_layers = 1
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)
        
        if(self.universal):  

            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        params =(hidden_size, 
                 total_key_depth or hidden_size,
                 total_value_depth or hidden_size,
                 filter_size, 
                 num_heads, 
                 _gen_bias_mask(max_length) if use_mask else None,
                 layer_dropout, 
                 attention_dropout, 
                 relu_dropout)
        
        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.embedding_proj1 = nn.Linear(embedding_size, hidden_size, bias=False)
        if(self.universal):
            self.enc_cr = EncoderLayer_cr(*params)
        else:
            self.enc_cr = nn.ModuleList([EncoderLayer_cr(*params) for _ in range(num_layers)])
        
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)
        self.input_dropout1 = nn.Dropout(input_dropout)
    def forward(self, inputs,inputs1, mask):

        x = self.input_dropout(inputs)
        x1 = self.input_dropout1(inputs1)

        x = self.embedding_proj(x)
        x1 = self.embedding_proj(x1)
        if(self.universal):
            if(self.args.act): 
                x, (self.remainders, self.n_updates) = self.act_fn(x, inputs, self.enc, self.timing_signal, self.position_signal, self.num_layers)
                y = self.layer_norm(x)
            else:
                x1 += self.timing_signal[:, :inputs1.shape[1], :].type_as(inputs1.data)
                x1 += self.position_signal[:, l, :].unsqueeze(1).repeat(1,inputs1.shape[1],1).type_as(inputs1.data)
                for l in range(self.num_layers):
                    x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
                    x += self.position_signal[:, l, :].unsqueeze(1).repeat(1,inputs.shape[1],1).type_as(inputs.data)
                    x1 += self.timing_signal[:, :inputs1.shape[1], :].type_as(inputs1.data)
                    x1 += self.position_signal[:, l, :].unsqueeze(1).repeat(1,inputs1.shape[1],1).type_as(inputs1.data)
                    x = self.enc_cr(x, x1, mask=mask)
                y = self.layer_norm(x)
        else:
            
            for i in range(self.num_layers):
                x = self.enc_cr[i](x, x1, mask=mask)
        
            y = self.layer_norm(x)
        return y

class Decoder(nn.Module):
    """
    A Transformer Decoder module. 
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self, args, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=1000, input_dropout=0.0, layer_dropout=0.0, 
                 attention_dropout=0.0, relu_dropout=0.0, universal=False):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """
        
        super(Decoder, self).__init__()
        self.args = args
        self.universal = universal
        self.num_layers = num_layers
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)
        
        if(self.universal):  
  
            self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        self.mask = _get_attn_subsequent_mask(self.args, max_length)

        params =(args,
                 hidden_size,
                 total_key_depth or hidden_size,
                 total_value_depth or hidden_size,
                 filter_size, 
                 num_heads, 
                 _gen_bias_mask(max_length),
                 layer_dropout, 
                 attention_dropout, 
                 relu_dropout)
        
        if(self.universal):
            self.dec = DecoderLayer(*params)
        else:
            self.dec = nn.Sequential(*[DecoderLayer(*params) for l in range(num_layers)])
        
        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)
        self.attn_loss = nn.MSELoss()

    def forward(self, inputs, encoder_output, mask=None, pred_emotion=None, emotion_contexts=None, context_vad=None):
        '''
        inputs: (bsz, tgt_len)
        encoder_output: (bsz, src_len), src_len=dialog_len+concept_len
        mask: (bsz, src_len)
        pred_emotion: (bdz, emotion_type)
        emotion_contexts: (bsz, emb_dim)
        context_vad: (bsz, src_len) emotion intensity values
        '''
        mask_src, mask_trg = mask
        dec_mask = torch.gt(mask_trg.bool() + self.mask[:, :mask_trg.size(-1), :mask_trg.size(-1)].bool(), 0)

        x = self.input_dropout(inputs)
        x = self.embedding_proj(x)
        loss_att = 0.0
        attn_dist = None
        if(self.universal):
            if(self.args.act):
                x, attn_dist, (self.remainders,self.n_updates) = self.act_fn(x, inputs, self.dec, self.timing_signal, self.position_signal, self.num_layers, encoder_output, decoding=True)
                y = self.layer_norm(x)

            else:
                x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
                for l in range(self.num_layers):
                    x += self.position_signal[:, l, :].unsqueeze(1).repeat(1,inputs.shape[1],1).type_as(inputs.data)
                    x, _, pred_emotion, emotion_contexts, attn_dist, _ = self.dec((x, encoder_output, pred_emotion, emotion_contexts, [], (mask_src,dec_mask)))
                y = self.layer_norm(x)
        else:

            x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)

            y, _, pred_emotion, emotion_contexts, attn_dist, _ = self.dec((x, encoder_output, pred_emotion, emotion_contexts, [], (mask_src,dec_mask)))

            if context_vad is not None:
                src_attn_dist = torch.mean(attn_dist, dim=1) 
                loss_att = self.attn_loss(src_attn_dist, context_vad)

            y = self.layer_norm(y)

        return y, attn_dist, loss_att

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, args, d_model, vocab):
        super(Generator, self).__init__()
        self.args = args
        self.proj = nn.Linear(d_model, vocab)
        self.emo_proj = nn.Linear(2 * d_model, vocab)
        self.p_gen_linear = nn.Linear(self.args.hidden_dim, 1)

    def forward(self, x, pred_emotion=None, emotion_context=None, attn_dist=None, enc_batch_extend_vocab=None, extra_zeros=None, temp=1):

        if self.args.pointer_gen:
            p_gen = self.p_gen_linear(x)
            alpha = torch.sigmoid(p_gen)

        if emotion_context is not None:

            pred_emotion = pred_emotion.repeat(1, x.size(1), 1)
            x = torch.cat((x, pred_emotion), dim=2)
            logit = self.emo_proj(x)
        else:
            logit = self.proj(x) 

        if self.args.pointer_gen:
            vocab_dist = F.softmax(logit/temp, dim=2)
            vocab_dist_ = alpha * vocab_dist

            attn_dist = F.softmax(attn_dist/temp, dim=-1)
            attn_dist_ = (1 - alpha) * attn_dist
            enc_batch_extend_vocab_ = torch.cat([enc_batch_extend_vocab.unsqueeze(1)]*x.size(1),1) 

            if extra_zeros is not None:
                extra_zeros = torch.cat([extra_zeros.unsqueeze(1)] * x.size(1), 1)
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 2)

            logit = torch.log(vocab_dist_.scatter_add(2, enc_batch_extend_vocab_, attn_dist_) + 1e-18)
            return logit
        else:
            return F.log_softmax(logit, dim=-1)


class INFEREM(nn.Module):
    def __init__(self, args, vocab, decoder_number,  model_file_path=None, is_eval=False, load_optim=False):
        super(INFEREM, self).__init__()
        self.args = args
        self.vocab = vocab
        word2index, word2count, index2word, n_words = vocab
        self.word2index = word2index
        self.word2count = word2count
        self.index2word = index2word
        self.vocab_size = n_words

        self.embedding = share_embedding(args, n_words, word2index, self.args.pretrain_emb) 



        self.encoder = Encoder(args, self.args.emb_dim, self.args.hidden_dim, num_layers=self.args.hop,
                               num_heads=self.args.heads, total_key_depth=self.args.depth, total_value_depth=self.args.depth,
                               max_length=args.max_seq_length, filter_size=self.args.filter, universal=self.args.universal)

        self.encoder1 = Encoder(args, self.args.emb_dim, self.args.hidden_dim, num_layers=self.args.hop,
                               num_heads=self.args.heads, total_key_depth=self.args.depth, total_value_depth=self.args.depth,
                               max_length=args.max_seq_length, filter_size=self.args.filter, universal=self.args.universal)


        self.encoder2 = Encoder(args, self.args.emb_dim, self.args.hidden_dim, num_layers=self.args.hop,
                               num_heads=self.args.heads, total_key_depth=self.args.depth, total_value_depth=self.args.depth,
                               max_length=args.max_seq_length, filter_size=self.args.filter, universal=self.args.universal)


        self.encoder3 = Encoder(args, self.args.emb_dim, self.args.hidden_dim, num_layers=self.args.hop,
                               num_heads=self.args.heads, total_key_depth=self.args.depth, total_value_depth=self.args.depth,
                               max_length=args.max_seq_length, filter_size=self.args.filter, universal=self.args.universal)
        

        self.encoder3 = Encoder(args, self.args.emb_dim, self.args.hidden_dim, num_layers=self.args.hop,
                               num_heads=self.args.heads, total_key_depth=self.args.depth, total_value_depth=self.args.depth,
                               max_length=args.max_seq_length, filter_size=self.args.filter, universal=self.args.universal)



        self.encoder_cr = Encoder_cr(args, self.args.emb_dim, self.args.hidden_dim, num_layers=1,
                               num_heads=self.args.heads, total_key_depth=self.args.depth, total_value_depth=self.args.depth,
                               max_length=args.max_seq_length, filter_size=self.args.filter, universal=self.args.universal)
        self.encoder_cr1 = Encoder_cr(args, self.args.emb_dim, self.args.hidden_dim, num_layers=self.args.hop,
                               num_heads=self.args.heads, total_key_depth=self.args.depth, total_value_depth=self.args.depth,
                               max_length=args.max_seq_length, filter_size=self.args.filter, universal=self.args.universal)

        self.map_emo = {0: 'surprised', 1: 'excited', 2: 'annoyed', 3: 'proud',
                        4: 'angry', 5: 'sad', 6: 'grateful', 7: 'lonely', 8: 'impressed',
                        9: 'afraid', 10: 'disgusted', 11: 'confident', 12: 'terrified',
                        13: 'hopeful', 14: 'anxious', 15: 'disappointed', 16: 'joyful',
                        17: 'prepared', 18: 'guilty', 19: 'furious', 20: 'nostalgic',
                        21: 'jealous', 22: 'anticipating', 23: 'embarrassed', 24: 'content',
                        25: 'devastated', 26: 'sentimental', 27: 'caring', 28: 'trusting',
                        29: 'ashamed', 30: 'apprehensive', 31: 'faithful'}

        self.dropout = args.dropout
        self.W_q = nn.Linear(args.emb_dim, args.emb_dim)
        self.W_k = nn.Linear(args.emb_dim, args.emb_dim)
        self.W_v = nn.Linear(args.emb_dim, args.emb_dim)
        self.graph_out = nn.Linear(args.emb_dim, args.emb_dim)
        self.graph_layer_norm = LayerNorm(args.hidden_dim)

        self.identify = nn.Linear(args.emb_dim, decoder_number, bias=False)
        self.activation = nn.Softmax(dim=1)

        self.emotion_embedding = nn.Linear(decoder_number, args.emb_dim)
        self.decoder = Decoder(args, args.emb_dim, hidden_size=args.hidden_dim,  num_layers=args.hop, num_heads=args.heads,
                               total_key_depth=args.depth,total_value_depth=args.depth, filter_size=args.filter, max_length=args.max_seq_length,)
        
        self.decoder_key = nn.Linear(args.hidden_dim, decoder_number, bias=False)
        self.generator = Generator(args, args.hidden_dim, self.vocab_size)

        if args.projection:
            self.embedding_proj_in = nn.Linear(args.emb_dim, args.hidden_dim, bias=False)
        if args.weight_sharing:
            self.generator.proj.weight = self.embedding.lut.weight

        self.criterion = nn.NLLLoss(ignore_index=args.PAD_idx)
        if args.label_smoothing:
            self.criterion = LabelSmoothing(size=self.vocab_size, padding_idx=args.PAD_idx, smoothing=0.1)
            self.criterion_ppl = nn.NLLLoss(ignore_index=args.PAD_idx)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)
        if args.noam:

            self.optimizer = NoamOpt(args.hidden_dim, 1, 8000, torch.optim.Adam(self.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

        if model_file_path is not None:
            print("loading weights")
            state = torch.load(model_file_path, map_location=lambda storage, location: storage)
            self.encoder.load_state_dict(state['encoder_state_dict'])
            self.encoder1.load_state_dict(state['encoder_state_dict'])

            self.decoder.load_state_dict(state['decoder_state_dict'])
            self.generator.load_state_dict(state['generator_dict'])
            self.embedding.load_state_dict(state['embedding_dict'])
            self.decoder_key.load_state_dict(state['decoder_key_state_dict']) 
            if load_optim:
                self.optimizer.load_state_dict(state['optimizer'])
            self.eval()

        self.model_dir = args.save_path
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.best_path = ""

    def save_model(self, running_avg_ppl, iter, f1_g,f1_b,ent_g,ent_b):
        state = {
            'iter': iter,
            'encoder_state_dict': self.encoder.state_dict(),
            'encoder_state_dict_lu': self.encoder_lu.state_dict(),            
            'decoder_state_dict': self.decoder.state_dict(),
            'generator_dict': self.generator.state_dict(),
            'decoder_key_state_dict': self.decoder_key.state_dict(),
            'embedding_dict': self.embedding.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'current_loss': running_avg_ppl
        }
        model_save_path = os.path.join(self.model_dir, 'model_{}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}.tar'.format(iter,running_avg_ppl,f1_g,f1_b,ent_g,ent_b) )
        self.best_path = model_save_path
        torch.save(state, model_save_path)

    def concept_graph(self, context, concept, adjacency_mask):
        '''

        :param context: (bsz, max_context_len, embed_dim)
        :param concept: (bsz, max_concept_len, embed_dim)
        :param adjacency_mask: (bsz, max_context_len, max_context_len + max_concpet_len)
        :return:
        '''

        target = context

        src = torch.cat((target, concept), dim=1) 

        q = self.W_q(target) 
        k, v = self.W_k(src), self.W_v(src)  
        attn_weights_ori = torch.bmm(q, k.transpose(1, 2)) 

        adjacency_mask = adjacency_mask.bool()
        attn_weights_ori.masked_fill_(
            adjacency_mask,
            1e-24
        )  
        attn_weights = torch.softmax(attn_weights_ori, dim=-1) 

        if torch.isnan(attn_weights).sum() != 0:
            pdb.set_trace()

        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)  
        attn = self.graph_out(attn)

        attn = F.dropout(attn, p=self.dropout, training=self.training)
        new_context = self.graph_layer_norm(target + attn)

        new_context = torch.cat((new_context, concept), dim=1)
        return new_context


    def concept_graph_lu(self, context, concept, adjacency_mask):
        '''

        :param context: (bsz, max_context_len, embed_dim)
        :param concept: (bsz, max_concept_len, embed_dim)
        :param adjacency_mask: (bsz, max_context_len, max_context_len + max_concpet_len)
        :return:
        '''
        target = context
        print('hhhhhhhhh')
        print(target.size())
        print(target)
        src = torch.cat((target, concept), dim=1)

        q = self.W_q(target) 
        k, v = self.W_k(src), self.W_v(src) 
        attn_weights_ori = torch.bmm(q, k.transpose(1, 2)) 

        adjacency_mask = adjacency_mask.bool()
        attn_weights_ori.masked_fill_(
            adjacency_mask,
            1e-24
        ) 
        attn_weights = torch.softmax(attn_weights_ori, dim=-1) 
        if torch.isnan(attn_weights).sum() != 0:
            pdb.set_trace()

        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_weights, v) 
        attn = self.graph_out(attn)

        attn = F.dropout(attn, p=self.dropout, training=self.training)
        new_context = self.graph_layer_norm(target + attn)

        new_context = torch.cat((new_context, concept), dim=1)
        return new_context

    def train_one_batch(self, batch, iter, train=True):
        enc_batch_lsu = batch["context_batch_lsu"]  
        enc_batch_lsup = batch["context_batch_lsup"]

        enc_batch_extend_vocab_lsu = batch["context_ext_batch_lsu"]
        enc_batch_extend_vocab_lsup = batch["context_ext_batch_lsup"]        

        enc_vad_batch_lsu = batch['context_vad_lsu']
        enc_vad_batch_lsup = batch['context_vad_lsup']

        concept_input_lsu = batch["concept_batch_lsu"]
        concept_input_lsup = batch["concept_batch_lsup"]

        concept_ext_input_lsu = batch["concept_ext_batch_lsu"]
        concept_vad_batch_lsu = batch['concept_vad_batch_lsu']
        concept_ext_input_lsup = batch["concept_ext_batch_lsup"]
        concept_vad_batch_lsup = batch['concept_vad_batch_lsup']

        oovs = batch["oovs"]
        max_oov_length = len(sorted(oovs, key=lambda i: len(i), reverse=True)[0])

        extra_zeros_lsu = Variable(torch.zeros((enc_batch_lsu.size(0), max_oov_length))).to(self.args.device)
        extra_zeros_lsup = Variable(torch.zeros((enc_batch_lsup.size(0), max_oov_length))).to(self.args.device)


        dec_batch_lsu = batch["target_batch_lsu"]

        dec_ext_batch_lsu = batch["target_ext_batch_lsu"]

        if self.args.noam:
            self.optimizer.optimizer.zero_grad()
        else:
            self.optimizer.zero_grad()


        mask_src_lsu = enc_batch_lsu.data.eq(self.args.PAD_idx).unsqueeze(1)
        mask_src_lsup = enc_batch_lsup.data.eq(self.args.PAD_idx).unsqueeze(1)

        emb_mask_lsu = self.embedding(batch["mask_context_lsu"]) 
        emb_mask_lsup = self.embedding(batch["mask_context_lsup"]) 

        src_emb_lsu = self.embedding(enc_batch_lsu)+emb_mask_lsu
        src_emb_lsup = self.embedding(enc_batch_lsup)+emb_mask_lsup

        src_vad_lsu = enc_vad_batch_lsu
        src_vad_lsup = enc_vad_batch_lsup

        if self.args.model != 'wo_ECE':
            if concept_input_lsu.size()[0] != 0:
                mask_con_lsu = concept_input_lsu.data.eq(self.args.PAD_idx).unsqueeze(1) 
                con_mask_lsu = self.embedding(batch["mask_concept_lsu"]) 
                con_emb_lsu = self.embedding(concept_input_lsu)+con_mask_lsu


                src_emb_lsu = self.concept_graph(src_emb_lsu, con_emb_lsu, batch["adjacency_mask_batch_lsu"]) 

                src_vad_lsu = torch.cat((enc_vad_batch_lsu, concept_vad_batch_lsu), dim=1)

        if self.args.model != 'wo_ECE': 
            if concept_input_lsup.size()[0] != 0:
                mask_con_lsup = concept_input_lsup.data.eq(self.args.PAD_idx).unsqueeze(1) 
                con_mask_lsup = self.embedding(batch["mask_concept_lsup"]) 
                con_emb_lsup = self.embedding(concept_input_lsup)+con_mask_lsup


                src_emb_lsup = self.concept_graph(src_emb_lsup, con_emb_lsup, batch["adjacency_mask_batch_lsup"]) 
                mask_src_lsup = torch.cat((mask_src_lsup, mask_con_lsup), dim=2) 
               
                src_vad_lsup = torch.cat((enc_vad_batch_lsup, concept_vad_batch_lsup), dim=1) 

        encoder_outputs_lsup = self.encoder(src_emb_lsup, mask_src_lsup)

        encoder_outputs_lsu = self.encoder1(src_emb_lsu, mask_src_lsu)
        encoder_outputs_cr_lsu = self.encoder_cr(encoder_outputs_lsup, encoder_outputs_lsu, mask_src_lsu)

        src_vad_lsu = torch.softmax(src_vad_lsu, dim=-1)
        src_vad_lsup = torch.softmax(src_vad_lsup, dim=-1)

        emotion_context_vad_lsup = src_vad_lsup.unsqueeze(2)
        emotion_context_vad_lsup = emotion_context_vad_lsup.repeat(1, 1, self.args.emb_dim) 
        emotion_context_lsup = torch.sum(emotion_context_vad_lsup * encoder_outputs_lsup, dim=1) 
        emotion_contexts_lsup = emotion_context_vad_lsup * encoder_outputs_lsup

        emotion_logit_lsup = self.identify(emotion_context_lsup)

        sos_emb_lsu = self.emotion_embedding(emotion_logit_lsup).unsqueeze(1)
        dec_emb_lsu = self.embedding(dec_batch_lsu[:, :-1]) 
        dec_emb_lsu = torch.cat((sos_emb_lsu, dec_emb_lsu), dim=1) 

        mask_trg_lsu = dec_batch_lsu.data.eq(self.args.PAD_idx).unsqueeze(1)


        if "wo_EDD" in self.args.model:
            pre_logit_lsu, attn_dist_lsu, loss_attn_lsu = self.decoder(inputs=dec_emb_lsu,
                                                           encoder_output=encoder_outputs_cr_lsu,
                                                           mask=(mask_src_lsup, mask_trg_lsu),
                                                           pred_emotion=None,
                                                           emotion_contexts=None)
        else:
            pre_logit_lsu, attn_dist_lsu, loss_attn_lsu = self.decoder(inputs=dec_emb_lsu,
                                                           encoder_output=encoder_outputs_cr_lsu,
                                                           mask=(mask_src_lsup,mask_trg_lsu),
                                                           pred_emotion=None,
                                                           emotion_contexts=emotion_context_lsup,
                                                           context_vad=src_vad_lsup)


        if self.args.model != 'wo_ECE': 
            if concept_input_lsup.size()[0] != 0:
                enc_batch_extend_vocab_lsup = torch.cat((enc_batch_extend_vocab_lsup, concept_ext_input_lsup), dim=1)  
            
        logit_lsu = self.generator(pre_logit_lsu, None, None, attn_dist_lsu, enc_batch_extend_vocab_lsup if self.args.pointer_gen else None, extra_zeros_lsup)

        _, decoded_words_lsu = torch.max(logit_lsu, dim = 2)

     
        loss_lsu = self.criterion(logit_lsu.contiguous().view(-1, logit_lsu.size(-1)),
                              dec_batch_lsu.contiguous().view(-1) if self.args.pointer_gen else dec_ext_batch_lsu.contiguous().view(-1))

        yon_batch = batch["yon_batch"]
    

        for i in range(len(yon_batch)):
            if yon_batch[i] == 0:
                for j in range(len(decoded_words_lsu[i])):
                    decoded_words_lsu[i][j] = 1

        decoded_words_lsu_emb = self.embedding(decoded_words_lsu)
        decoded_words_lsu_mask = decoded_words_lsu.data.eq(self.args.PAD_idx).unsqueeze(1)
        encoder_outputs_app = self.encoder2(decoded_words_lsu_emb, decoded_words_lsu_mask)

        enc_batch = batch["context_batch"]
        enc_batch_lu = batch["context_batch_lu"]       

 

        enc_batch_extend_vocab = batch["context_ext_batch"]
        enc_batch_extend_vocab_lu = batch["context_ext_batch_lu"]


        enc_vad_batch = batch['context_vad']
        enc_vad_batch_lu = batch['context_vad_lu']


        concept_input = batch["concept_batch"]  
        concept_input_lu = batch["concept_batch_lu"] 

        concept_ext_input = batch["concept_ext_batch"]
        concept_vad_batch = batch['concept_vad_batch']
        concept_ext_input_lu = batch["concept_ext_batch_lu"]
        concept_vad_batch_lu = batch['concept_vad_batch_lu']      

        extra_zeros = Variable(torch.zeros((enc_batch.size(0), max_oov_length))).to(self.args.device)
        extra_zeros_lu = Variable(torch.zeros((enc_batch_lu.size(0), max_oov_length))).to(self.args.device)

        dec_batch = batch["target_batch"]

        dec_ext_batch = batch["target_ext_batch"]

        mask_src = enc_batch.data.eq(self.args.PAD_idx).unsqueeze(1) 
        mask_src_lu = enc_batch_lu.data.eq(self.args.PAD_idx).unsqueeze(1)
  


        emb_mask = self.embedding(batch["mask_context"]) 
        emb_mask_lu = self.embedding(batch["mask_context_lu"])

    

        src_emb = self.embedding(enc_batch)+emb_mask
        src_emb_lu = self.embedding(enc_batch_lu)+emb_mask_lu

    
     
        src_vad = enc_vad_batch  
        src_vad_lu = enc_vad_batch_lu



        if self.args.model != 'wo_ECE': 
            if concept_input.size()[0] != 0:
                mask_con = concept_input.data.eq(self.args.PAD_idx).unsqueeze(1) 
                con_mask = self.embedding(batch["mask_concept"]) 
                con_emb = self.embedding(concept_input)+con_mask


                src_emb = self.concept_graph(src_emb, con_emb, batch["adjacency_mask_batch"])  
                mask_src = torch.cat((mask_src, mask_con), dim=2) 
        
                src_vad = torch.cat((enc_vad_batch, concept_vad_batch), dim=1) 

        if self.args.model != 'wo_ECE': 
            if concept_input_lu.size()[0] != 0:
                mask_con_lu = concept_input_lu.data.eq(self.args.PAD_idx).unsqueeze(1) 
                con_mask_lu = self.embedding(batch["mask_concept_lu"]) 
                con_emb_lu = self.embedding(concept_input_lu)+con_mask_lu


                src_emb_lu = self.concept_graph(src_emb_lu, con_emb_lu, batch["adjacency_mask_batch_lu"]) 
                mask_src_lu = torch.cat((mask_src_lu, mask_con_lu), dim=2) 
        
                src_vad_lu = torch.cat((enc_vad_batch_lu, concept_vad_batch_lu), dim=1) 

        encoder_outputs = self.encoder(src_emb, mask_src) 
        encoder_outputs_lu = self.encoder1(src_emb_lu, mask_src_lu)  

        encoder_outputs_lu_final = torch.cat((encoder_outputs_app, encoder_outputs_lu), dim=1) 
        mask_src_lu_final = torch.cat((decoded_words_lsu_mask, mask_src_lu), dim=2)     
         
        encoder_outputs_lu_final = self.encoder3(encoder_outputs_lu_final, mask_src_lu_final) 


        encoder_outputs_cr = self.encoder_cr(encoder_outputs,  encoder_outputs_lu_final, mask_src_lu_final)

        src_vad = torch.softmax(src_vad, dim=-1)
        src_vad_lu = torch.softmax(src_vad_lu, dim=-1)


        emotion_context_vad = src_vad.unsqueeze(2)
        emotion_context_vad = emotion_context_vad.repeat(1, 1, self.args.emb_dim)
        emotion_context = torch.sum(emotion_context_vad * encoder_outputs, dim=1)
        emotion_contexts = emotion_context_vad * encoder_outputs





        emotion_logit = self.identify(emotion_context)
        loss_emotion = nn.CrossEntropyLoss(reduction='sum')(emotion_logit, batch['emotion_label'])


        pred_emotion = np.argmax(emotion_logit.detach().cpu().numpy(), axis=1)
        emotion_acc = accuracy_score(batch["emotion_label"].cpu().numpy(), pred_emotion)

        sos_emb = self.emotion_embedding(emotion_logit).unsqueeze(1) 
        dec_emb = self.embedding(dec_batch[:, :-1]) 
        dec_emb = torch.cat((sos_emb, dec_emb), dim=1) 

       

        mask_trg = dec_batch.data.eq(self.args.PAD_idx).unsqueeze(1)
 
        if "wo_EDD" in self.args.model:
            pre_logit, attn_dist, loss_attn = self.decoder(inputs=dec_emb,
                                                           encoder_output=encoder_outputs_cr,
                                                           mask=(mask_src, mask_trg),
                                                           pred_emotion=None,
                                                           emotion_contexts=None)
        else:
            pre_logit, attn_dist, loss_attn = self.decoder(inputs=dec_emb,
                                                           encoder_output=encoder_outputs_cr,
                                                           mask=(mask_src,mask_trg),
                                                           pred_emotion=None,
                                                           emotion_contexts=emotion_context,
                                                           context_vad=src_vad)

        if self.args.model != 'wo_ECE':  
            if concept_input.size()[0] != 0:
                enc_batch_extend_vocab = torch.cat((enc_batch_extend_vocab, concept_ext_input), dim=1)

        logit = self.generator(pre_logit, None, None, attn_dist, enc_batch_extend_vocab if self.args.pointer_gen else None, extra_zeros_lu)
        _, decoded_words = torch.max(logit, dim = 2)
        
        loss = self.criterion(logit.contiguous().view(-1, logit.size(-1)),
                              dec_batch.contiguous().view(-1) if self.args.pointer_gen else dec_ext_batch.contiguous().view(-1))

        if loss_lsu > loss:
            loss += (1.5*loss_lsu) 
        else:
            loss += (0.3*loss_lsu)    
            
        loss += (1.2*loss_emotion)
        if self.args.attn_loss and self.args.model != "wo_EDD":

            loss += (0.12 * loss_attn)

        loss_ppl = 0.0
        if self.args.label_smoothing:
            loss_ppl = self.criterion_ppl(logit.contiguous().view(-1, logit.size(-1)),
                                          dec_batch.contiguous().view(-1) if self.args.pointer_gen else dec_ext_batch.contiguous().view(-1)).item()

        if torch.sum(torch.isnan(loss)) != 0:
            print('loss is NAN :(')
            pdb.set_trace()

        if train:
            loss.backward()
            self.optimizer.step()

        if self.args.label_smoothing:
            return loss_ppl, math.exp(min(loss_ppl, 100)), loss_emotion.item(), emotion_acc
        else:
            return loss.item(), math.exp(min(loss.item(), 100)), 0, 0

    def compute_act_loss(self,module):    
        R_t = module.remainders
        N_t = module.n_updates
        p_t = R_t + N_t
        avg_p_t = torch.sum(torch.sum(p_t,dim=1)/p_t.size(1))/p_t.size(0)
        loss = self.args.act_loss_weight * avg_p_t.item()
        return loss

    def decoder_greedy(self, batch, max_dec_step=30):

        enc_batch_extend_vocab_lsu, extra_zeros_lsu = None, None
        enc_batch_lsup = batch["context_batch_lsup"]
        enc_batch_lsu = batch["context_batch_lsu"]

        enc_vad_batch_lsup = batch['context_vad_lsup']
        enc_vad_batch_lsu = batch['context_vad_lsu']

        enc_batch_extend_vocab_lsup = batch["context_ext_batch_lsup"]
        enc_batch_extend_vocab_lsu = batch["context_ext_batch_lsu"]

        concept_input_lsup = batch["concept_batch_lsup"] 
        concept_ext_input_lsup = batch["concept_ext_batch_lsup"]
        concept_vad_batch_lsup = batch['concept_vad_batch_lsup']

        concept_input_lsu = batch["concept_batch_lsu"]
        concept_ext_input_lsu = batch["concept_ext_batch_lsu"]
        concept_vad_batch_lsu = batch['concept_vad_batch_lsu']


        oovs_lsu = batch["oovs"]
        max_oov_length_lsu = len(sorted(oovs_lsu, key=lambda i: len(i), reverse=True)[0])
        extra_zeros_lsu = Variable(torch.zeros((enc_batch_lsup.size(0), max_oov_length_lsu))).to(self.args.device)

        mask_src_lsup = enc_batch_lsup.data.eq(self.args.PAD_idx).unsqueeze(1) 
        emb_mask_lsup = self.embedding(batch["mask_context_lsup"])
        src_emb_lsup = self.embedding(enc_batch_lsup) + emb_mask_lsup
        src_vad_lsup = enc_vad_batch_lsup 

        mask_src_lsu = enc_batch_lsu.data.eq(self.args.PAD_idx).unsqueeze(1) 
        emb_mask_lsu = self.embedding(batch["mask_context_lsu"])
        src_emb_lsu = self.embedding(enc_batch_lsu) + emb_mask_lsu
        src_vad_lsu = enc_vad_batch_lsu 


        if self.args.model != 'wo_ECE': 
            if concept_input_lsup.size()[0] != 0:
                mask_con_lsup = concept_input_lsup.data.eq(self.args.PAD_idx).unsqueeze(1) 
                con_mask_lsup = self.embedding(batch["mask_concept_lsup"]) 
                con_emb_lsup = self.embedding(concept_input_lsup) + con_mask_lsup

                src_emb_lsup = self.concept_graph(src_emb_lsup, con_emb_lsup,
                                             batch["adjacency_mask_batch_lsup"]) 
                mask_src_lsup = torch.cat((mask_src_lsup, mask_con_lsup), dim=2)

                src_vad_lsup = torch.cat((enc_vad_batch_lsup, concept_vad_batch_lsup), dim=1)


        if self.args.model != 'wo_ECE':
            if concept_input_lsu.size()[0] != 0:
                mask_con_lsu = concept_input_lsu.data.eq(self.args.PAD_idx).unsqueeze(1)
                con_mask_lsu = self.embedding(batch["mask_concept_lsu"])
                con_emb_lsu = self.embedding(concept_input_lsu) + con_mask_lsu

                src_emb_lsu = self.concept_graph(src_emb_lsu, con_emb_lsu,
                                             batch["adjacency_mask_batch_lsu"]) 
                mask_src_lsu = torch.cat((mask_src_lsu, mask_con_lsu), dim=2) 

                src_vad_lsu = torch.cat((enc_vad_batch_lsu, concept_vad_batch_lsu), dim=1) 

        encoder_outputs_lsup = self.encoder(src_emb_lsup, mask_src_lsup)

        encoder_outputs_lsu = self.encoder1(src_emb_lsu, mask_src_lsu)
        encoder_outputs_cr_lsu = self.encoder_cr(encoder_outputs_lsup, encoder_outputs_lsu, mask_src_lsu)

        src_vad_lsup = torch.softmax(src_vad_lsup, dim=-1)
        emotion_context_vad_lsup = src_vad_lsup.unsqueeze(2)
        emotion_context_vad_lsup = emotion_context_vad_lsup.repeat(1, 1, self.args.emb_dim) 
        emotion_context_lsup = torch.sum(emotion_context_vad_lsup * encoder_outputs_lsup, dim=1) 

        emotion_logit_lsup = self.identify(emotion_context_lsup) 

        if concept_input_lsup.size()[0] != 0 and self.args.model != 'wo_ECE':
            enc_ext_batch_lsup = torch.cat((enc_batch_extend_vocab_lsup, concept_ext_input_lsup), dim=1)
        else:
            enc_ext_batch_lsup = enc_batch_extend_vocab_lsup

        ys_lsup = torch.ones(1, 1).fill_(self.args.SOS_idx).long()
        ys_emb_lsup = self.emotion_embedding(emotion_logit_lsup).unsqueeze(1) 
        sos_em_lsup = ys_emb_lsup  
        if self.args.USE_CUDA:
            ys_lsup = ys_lsup.cuda()
        mask_trg_lsup = ys_lsup.data.eq(self.args.PAD_idx).unsqueeze(1)
        decoded_words_lsup = []
        decoded_indexs_lsup = []
        for i in range(max_dec_step+1):
            if self.args.projection:
                out_lsup, attn_dist_lsup, _ = self.decoder(self.embedding_proj_in(ys_emb_lsup), self.embedding_proj_in(encoder_outputs_cr_lsu), (mask_src_lsup,mask_trg_lsup))
            else:
                out_lsup, attn_dist_lsup, _ = self.decoder(inputs=ys_emb_lsup,
                                                 encoder_output=encoder_outputs_cr_lsu,
                                                 mask=(mask_src_lsup,mask_trg_lsup),
                                                 pred_emotion=None,
                                                 emotion_contexts=emotion_context_lsup,
                                                 context_vad=src_vad_lsup)

            prob_lsup = self.generator(out_lsup, None, None, attn_dist_lsup, enc_ext_batch_lsup if self.args.pointer_gen else None, extra_zeros_lsu)
            _, next_word_lsup = torch.max(prob_lsup[:, -1], dim = 1)
            decoded_indexs_lsup.append(next_word_lsup.view(-1).item())
            decoded_words_lsup.append(['<EOS>' if ni.item() == self.args.EOS_idx else self.index2word[str(ni.item())] for ni in next_word_lsup.view(-1)])
            next_word_lsup = next_word_lsup.data[0]

            if self.args.use_cuda:
                ys_lsup = torch.cat([ys_lsup, torch.ones(1, 1).long().fill_(next_word_lsup).cuda()], dim=1)
                ys_lsup = ys_lsup.cuda()
                ys_emb_lsup = torch.cat((ys_emb_lsup, self.embedding(torch.ones(1, 1).long().fill_(next_word_lsup).cuda())), dim=1)

            mask_trg_lsup = ys_lsup.data.eq(self.args.PAD_idx).unsqueeze(1)

        sent_lsup = []
        for _, row in enumerate(np.transpose(decoded_words_lsup)):
            st = ''
            for e in row:
                if e == '<EOS>': break
                else: st+= e + ' '
            sent_lsup.append(st)
            

        decoded_indexs1_lsup = []


        for i in range(len(decoded_indexs_lsup)):
            if decoded_indexs_lsup[i] != 2:
                decoded_indexs1_lsup.append(decoded_indexs_lsup[i])
            else:
                break
       
        yon_batch = batch["yon_batch"]

        if yon_batch[0] == 0:
            for j in range(len(decoded_indexs1_lsup)):
                decoded_indexs1_lsup[j] = 1
            

        decoded_indexs1_lsup = [decoded_indexs1_lsup]
        decoded_indexs1_lsup = torch.tensor(decoded_indexs1_lsup)
        decoded_indexs1_lsup = decoded_indexs1_lsup.cuda()        

        decoded_words_lsu_emb = self.embedding(decoded_indexs1_lsup)
        decoded_words_lsu_mask = decoded_indexs1_lsup.data.eq(self.args.PAD_idx).unsqueeze(1)
        encoder_outputs_app = self.encoder2(decoded_words_lsu_emb, decoded_words_lsu_mask)


        enc_batch_extend_vocab, extra_zeros = None, None
        enc_batch = batch["context_batch"]
        enc_batch_lu = batch["context_batch_lu"]

        enc_vad_batch = batch['context_vad']
        enc_vad_batch_lu = batch['context_vad_lu']

        enc_batch_extend_vocab = batch["context_ext_batch"]
        enc_batch_extend_vocab_lu = batch["context_ext_batch_lu"]

        concept_input = batch["concept_batch"]  
        concept_ext_input = batch["concept_ext_batch"]
        concept_vad_batch = batch['concept_vad_batch']

        concept_input_lu = batch["concept_batch_lu"]  
        concept_ext_input_lu = batch["concept_ext_batch_lu"]
        concept_vad_batch_lu = batch['concept_vad_batch_lu']


        oovs = batch["oovs"]
        max_oov_length = len(sorted(oovs, key=lambda i: len(i), reverse=True)[0])
        extra_zeros = Variable(torch.zeros((enc_batch.size(0), max_oov_length))).to(self.args.device)

        mask_src = enc_batch.data.eq(self.args.PAD_idx).unsqueeze(1)  
        emb_mask = self.embedding(batch["mask_context"])
        src_emb = self.embedding(enc_batch) + emb_mask
        src_vad = enc_vad_batch  

        mask_src_lu = enc_batch_lu.data.eq(self.args.PAD_idx).unsqueeze(1) 
        emb_mask_lu = self.embedding(batch["mask_context_lu"])
        src_emb_lu = self.embedding(enc_batch_lu) + emb_mask_lu
        src_vad_lu = enc_vad_batch_lu 


        if self.args.model != 'wo_ECE':  
            if concept_input.size()[0] != 0:
                mask_con = concept_input.data.eq(self.args.PAD_idx).unsqueeze(1)  
                con_mask = self.embedding(batch["mask_concept"]) 
                con_emb = self.embedding(concept_input) + con_mask

                src_emb = self.concept_graph(src_emb, con_emb,
                                             batch["adjacency_mask_batch"]) 
                mask_src = torch.cat((mask_src, mask_con), dim=2)  

                src_vad = torch.cat((enc_vad_batch, concept_vad_batch), dim=1) 


        if self.args.model != 'wo_ECE':  
            if concept_input_lu.size()[0] != 0:
                mask_con_lu = concept_input_lu.data.eq(self.args.PAD_idx).unsqueeze(1)  
                con_mask_lu = self.embedding(batch["mask_concept_lu"]) 
                con_emb_lu = self.embedding(concept_input_lu) + con_mask_lu

                src_emb_lu = self.concept_graph(src_emb_lu, con_emb_lu,
                                             batch["adjacency_mask_batch_lu"]) 
                mask_src_lu = torch.cat((mask_src_lu, mask_con_lu), dim=2) 

                src_vad_lu = torch.cat((enc_vad_batch_lu, concept_vad_batch_lu), dim=1) 


        encoder_outputs = self.encoder(src_emb, mask_src)  
        encoder_outputs_lu = self.encoder1(src_emb_lu, mask_src_lu)

        encoder_outputs_lu_final = torch.cat((encoder_outputs_app, encoder_outputs_lu), dim=1) 
        mask_src_lu_final = torch.cat((decoded_words_lsu_mask, mask_src_lu), dim=2) 

        encoder_outputs_lu_final = self.encoder3(encoder_outputs_lu_final, mask_src_lu_final) 


        encoder_outputs_cr = self.encoder_cr(encoder_outputs,  encoder_outputs_lu_final, mask_src_lu_final)


        src_vad = torch.softmax(src_vad, dim=-1)
        emotion_context_vad = src_vad.unsqueeze(2)
        emotion_context_vad = emotion_context_vad.repeat(1, 1, self.args.emb_dim) 
        emotion_context = torch.sum(emotion_context_vad * encoder_outputs, dim=1)
        emotion_contexts = emotion_context_vad * encoder_outputs

        emotion_logit = self.identify(emotion_context)

        if concept_input.size()[0] != 0 and self.args.model != 'wo_ECE':
            enc_ext_batch = torch.cat((enc_batch_extend_vocab, concept_ext_input), dim=1)
        else:
            enc_ext_batch = enc_batch_extend_vocab

        ys = torch.ones(1, 1).fill_(self.args.SOS_idx).long()
        ys_emb = self.emotion_embedding(emotion_logit).unsqueeze(1)
        sos_emb = ys_emb  
        if self.args.USE_CUDA:
            ys = ys.cuda()
        mask_trg = ys.data.eq(self.args.PAD_idx).unsqueeze(1)
        decoded_words = []
        for i in range(max_dec_step+1):
            if self.args.projection:
                out, attn_dist, _ = self.decoder(self.embedding_proj_in(ys_emb), self.embedding_proj_in(encoder_outputs), (mask_src,mask_trg))
            else:
                out, attn_dist, _ = self.decoder(inputs=ys_emb,
                                                 encoder_output=encoder_outputs_cr,
                                                 mask=(mask_src,mask_trg),
                                                 pred_emotion=None,
                                                 emotion_contexts=emotion_context,
                                                 context_vad=src_vad)

            prob = self.generator(out, None, None, attn_dist, enc_ext_batch if self.args.pointer_gen else None, extra_zeros)
            _, next_word = torch.max(prob[:, -1], dim = 1)
            decoded_words.append(['<EOS>' if ni.item() == self.args.EOS_idx else self.index2word[str(ni.item())] for ni in next_word.view(-1)])
            next_word = next_word.data[0]

            if self.args.use_cuda:
                ys = torch.cat([ys, torch.ones(1, 1).long().fill_(next_word).cuda()], dim=1)
                ys = ys.cuda()
                ys_emb = torch.cat((ys_emb, self.embedding(torch.ones(1, 1).long().fill_(next_word).cuda())), dim=1)
            else:
                ys = torch.cat([ys, torch.ones(1, 1).long().fill_(next_word)], dim=1)
                ys_emb = torch.cat((ys_emb, self.embedding(torch.ones(1, 1).long().fill_(next_word))), dim=1)
            mask_trg = ys.data.eq(self.args.PAD_idx).unsqueeze(1)

        sent = []
        for _, row in enumerate(np.transpose(decoded_words)):
            st = ''
            for e in row:
                if e == '<EOS>': break
                else: st+= e + ' '
            sent.append(st)
        return sent