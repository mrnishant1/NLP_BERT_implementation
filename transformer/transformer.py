import numpy as np
from .tensor import Tensor
from .nn import MLP

def sinusoidal_positional_encoding(m_words:int, n_dims:int):
    #number of words, n_dims = embedding dims of each word
    #P_E(pos,2i) = sin(pos/10000^(2i/n_dims))
    #P_E(pos,2i+1) = cos(pos/10000^(2i/n_dims))

    #division term = position/10000^(2i/n_dims) = position * 10000^-(2i/n_dims) = position * e^(-2*position*log(10000)/n_dims)
    position = np.arange(m_words)[:,np.newaxis]
    div_term = np.exp(-2*position*np.log(10000)/n_dims)

    pos_embed = np.zeros(shape=(m_words,n_dims))
    #even position
    pos_embed[:,0::2] = np.sin(position*div_term)
    pos_embed[:,1::2] = np.cos(position*div_term)
    pos_embed = Tensor(pos_embed,label='pos_embed')
    return pos_embed

def layer_normalize(matrix:list):
    # matrix = multihead attention embedding [mxn]
    constant = 0.001 # to preven from being zero at denominator
    l2 = np.array(matrix)
    mean = np.mean(l2,axis=1,keepdims=True)[:,np.newaxis] #returns [mX1]
    x_minus_mean = l2 - mean
    var = np.sqrt(np.mean(np.square(x_minus_mean),axis=1)[:,np.newaxis] + constant) #[mx1]

    normalized = x_minus_mean/var
    return normalized

class attention_head:
    def __init__(self, shape: tuple[int, int] = (1, 1)):
        self.m, self.n = shape
        self.W_q = Tensor(np.random.randn(self.n,self.n),label='w_q')
        self.W_k = Tensor(np.random.randn(self.n,self.n),label='w_k')
        self.W_v = Tensor(np.random.randn(self.n,self.n),label='w_v')

    def __call__(self, P_E:Tensor):
        # """Attention"""
        Q = P_E * self.W_q
        K = P_E * self.W_k
        V = P_E * self.W_v
        div_factor = Tensor(1/np.sqrt(self.n))
        similarity = div_factor.elementwise_mul(Q * K.transpose())
        context_embed:Tensor = (similarity * V).softmax() # softmax((1/sqrt(dk))*Q.K)*V

        # """Add and Norm"""
        context_embed = context_embed + P_E
        return context_embed
    def perameter(self):
        return self.W_q,self.W_k,self.W_v


class Transformer:
    def __init__(self,word_embeds:list,activation_fn, n_heads:int = 8, ):
        self.word_embeds = Tensor(word_embeds,label='Data ingestion')
        print("Data Ingestion Successfull, Shape: ", self.word_embeds.data.shape)
        self.m,self.n = np.shape(self.word_embeds.data)
        self.n_heads = n_heads #number of attention head
        self.W_multi_head = Tensor(np.random.randn(self.n*n_heads, self.n),label="multihead")
        self.all_attention_heads = [attention_head(shape=(self.m,self.n)) for _ in range(self.n_heads)]
        self.lr = 0.01
        # """ FFN Model """

    def encoder(self):
        pos_encode:Tensor = sinusoidal_positional_encoding(self.m,self.n)
        P_E = self.word_embeds.__add__(pos_encode)  #P_E = positional encoding

        # Multi head attention

        tensors = [head(P_E) for head in self.all_attention_heads]
        multi_head_attention:Tensor = Tensor.cat(tensors) * self.W_multi_head
        add = multi_head_attention + P_E
        add = add -add.mean()
        add_n_norm = add.elementwise_mul(add.std()**-1); add_n_norm.label = 'add n norm'


        # """ Feed Forward Neural Network """
        # model = self.model
        # FFNN1:list[Tensor] = model(add_n_norm)

        # add_n_norm2 = add_n_norm + Tensor.cat(FFNN1)
        # return add_n_norm2

    def initialse_model(self, activation_fn):
        valid_activations = {"tanh", "relu"}
        if activation_fn not in valid_activations:
            raise ValueError(f"activation_fn must be one of {valid_activations}")
        model = MLP(n_inp=self.n, Layer_outs=[64, 64, 32, self.n],activation_fn=activation_fn)
        return model

    def Perameter(self):
        params = [self.W_multi_head]
        for head in self.all_attention_heads:
            params.extend(head.perameter())
        params.extend(self.model.parameters())
        return params