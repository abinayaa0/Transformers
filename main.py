import torch
import torch.nn as nn
class selfattention(nn.Module):
  def __init__(self,embed_size,heads):
    super(selfattention,self).__init__()
    self.embed_size=embed_size  #dmodel
    self.heads=heads  #number of heads in a multi head dimension layer
    self.head_dim=embed_size//heads #dmodel/h

    assert(self.head_dim*heads==embed_size)  #the embeddings needs to be divisible by the heads.

    #The inputs to the self attention layer: values, keys and query and fc_out
    self.values=nn.Linear(embed_size,embed_size,bias=False)  #thiss is XW where W is trainable W is the weight matrix for V
    self.keys=nn.Linear(embed_size,embed_size,bias=False)   #similar
    self.queries=nn.Linear(embed_size,embed_size,bias=False)  #similar
    self.fc_out=nn.Linear(embed_size,embed_size,bias=False) #the concatenation of all head_inputs w_0

  def forward(self,values,keys,queries):

      '''Step 1: project them into new layers'''
      #dimensions of the query matrix=(N,query_len,embed_size)
      #N=number of training examples, query_len is the i, embed_size is the dmodel.
      N=query.shape[0] #number of training examples.
      value_len,key_len,query_len=values.shape[1],keys.shape[1],queries.shape[1]
      values=self.values(values)
      keys=self.keys(keys)
      queries=self.queries(queries)
      '''step 2: split the whole input vector into multiple heads'''
      #the length of key and values is always the same, in case of encoder the value of query , k and v are the same
      #but in case of decoder q can be different from k nad v
      #we know that embed_size=head_dim*number of heads.
      #split the embeddings into self head pieces: this is dividing the whole input into multiple heads explicitly
      values=values.reshape(N,value_len,self.heads,self.head_dim)
      keys=keys.reshape(N,keys_len,self.heads,self.head_dim)
      queries=queries.reshape(N,queries_len,self.heads,self.head_dim)
      
      '''step 3: QK^T  --> calculate einsum to get attention scores in higher dimensions'''
      #refer to the attention formula in the research paper.
      # say,shape of  Qk=(m_tokens, d) and K=(m_tokens,d) then QK^T works of dimension (m_tokens,m_tokens) but when we have  
      #When we add the batch Qk=(N,m_tokens, d) and K=(N,m_tokens,d) then the QKt doesnt work So what do we do? einsum
      #einsum is a more generalized function to do matrix multiplication
      attn_scores=torch.einsum("nqhd,nkhd->nhqk",[queries, keys]) 
      #mention the dimensions of the input and the dimensions that we want to map to.


      '''step 4:masking to handle padded data'''
      #whenever we do not have the vectors of same dimensions in a Q or K we pad them with 0s to make them of equal length.
      #if we have 0 in the attention score then this mask will replace it with a very large negative number 
      #so that when softmax is applied to it it becomes 0.
      #this way the model ensures that the model does not become noisy due to padding.

      if mask is not None:
        attn_scores=attn_scores.masked_fill(mask==0,float("-1e20"))

      '''step 5:scale and normalize'''
      attention=torch.softmax(attn_scores/(self.head_dim**(1/2)),dim=3)  #softmax((QK^T)/rootd)  refer paper
      #applying query token in dimension three tells us that for every query token there is a probability distribution over the keys.

      '''step 6: multiply the above with value matrix'''
      out=torch.einsum("nhql,nlhd->nqhd",[attention,values]).reshape(N,query_len,self.heads*self.head_dim)#reshaping it back to the input size
      #NOTE:the key length is always equal to the value length. so k=l.
      return out
class Transformerblock(nn.Module):
  def __init__(self,embed_size,heads,dropout,forward_expansion):
    super(Transformerblock,self).__init__()
    self.attention=selfattention(embed_size,heads)  #the output of selfattention is attention vector
    self.norm1=nn.LayerNorm(embed_size)  #normaliza it
    self.norm2=nn.LayerNorm(embed_size) #normalize it
    
    self.feed_forward=nn.Sequential(   #build a neural network by stacking other modules
        nn.Linear(embed_size,forward_expansion*embed_size),
        nn.ReLu(),
        nn.Linear(forward_expansion*embed_size,embed_size),

    )
    self.dropout=nn.Dropout(dropout)
  def forward(self,value,key,query,mask):
    attention=self.attention(value,key,query,mask)
    x=self.dropout(self.norm1(attention+query))
    forward=self.feed_forward(x)
    out=self.dropout(self.norm2(forward+x))
    return out
    

#this is one encoder block.multiple encoder blocks are  stacked together
class Encoder(nn.Module):
  def __init__(self,srs_vocabsize,embed_size,num_layers,heads,device,forward_expansion,dropout,max_length):
    super(Encoder,self).__init__()
    self.embed_size=embed_size
    self.device=device
    self.word_embedding=nn.Embedding(srs_vocabsize,embed_size)    #trainable dictionary lookup table where number of keys are the vocab size
    self.position_embedding=nn.Embedding(max_length,embed_size)  #trainable position embedding
    '''stack the encoder blocks on top of each other'''
    self.layers=nn.ModuleList(
        [
            Transformerblock(
                embed_size,
                heads,
                dropout,
                forward_expansion=forward_expansion,

            )
            for _ in range(num_layers)
        ]
    )
    self.dropout=nn.Dropout(dropout)  
  def forward(self,x,mask):
    N,seq_length=x.shape
    positions=torch.arange(0,seq_length).expand(N,seq_length).to(self.device)
    out=self.dropout(
        (self.word_embedding(x)+self.position_embedding(positions))
    )

    
    #in the encoder the query, key and value are the same.
    for layer in self.layers:
      out=layer(out,out,out,mask)   #layer processing current output out is passed as Q,K,V to the layer function as input along with mask.
    return out
class Decoderblock(nn.Module):
  def __init__(self,srs_vocabsize,embed_size,num_layers,heads,device,forward_expansion,dropout,max_length):
    super(Decoderblock,self).__init__()
    self.norm=nn.LayerNorm(embed_size)
    self.attention=nn.selfattention(embed_size,heads=heads)
    self.transformerblock=Transformerblock(embed_size,heads,dropout,forward_expansion)
    self.dropout=nn.Dropout(dropout)

  def forward(self,x,value, key,src_mask,tar_mask):
   attention=self.attention(x,x,x,tar_mask)
   query=self.dropout(self.norm(attention+x))
   out=self.transformerblock(value,key,query,src_mask)
   return out

class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device, max_length):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                Decoderblock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size) #embed_dim -> vocab_size
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)

        return out
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, embed_size=512, num_layers=6, forward_expansion=4, heads=8, dropout=0, device="cpu", max_length=100):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
        )

        self.src_pad_idx = src_pad_idx #pad tokens
        self.trg_pad_idx = trg_pad_idx #future token mask
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )

        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out
