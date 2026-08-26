from .backprop import Tensor

from typing import Literal
class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []
    
# Single Neuron with its own weights and bias based on number of inputs(n_inp)
class Neuron(Module):
    def __init__(self, n_inp, non_linear=True, activation_fn: Literal["tanh", "relu"] = "tanh"):
        valid_activations = {"tanh", "relu"}

        if activation_fn not in valid_activations:
            raise ValueError(
                f"activation_fn must be one of {valid_activations}"
            )

        self.W = Tensor(
            [[np.random.uniform(-1, 1) for _ in range(n_inp)]],
            label="w"
        )
        self.b = Tensor([[0.0]], label="bias")
        self.non_linear = non_linear
        self.activation_fn = activation_fn

    def __call__(self, x):
        if isinstance(x, list) and all(isinstance(xi, Tensor) for xi in x):
            x = Tensor.cat(x, axis=1).transpose()   # previous layer's outputs -> (n_inp, 1)
        elif isinstance(x,Tensor):
            x = x.transpose() 
        else:
            x = Tensor(np.atleast_2d(x).transpose())   # raw numeric input -> (n_inp, 1)
        act = (self.W * x) + self.b     
        act = act.transpose()
        if not self.non_linear: #Linear actication 
            return act
        activation = getattr(act, self.activation_fn) #Non Linear actication
        return activation()

    def parameters(self):
        return [self.W, self.b]

    def __repr__(self):
        return f"{'Tanh' if self.non_linear else 'Linear'} Neuron({self.W.data.shape[1]})"

    
# Layer contains  
class Layer(Module):
    def __init__(self,n_inp, n_out,activation_fn: Literal["tanh", "relu"] = "tanh", non_linear=True, **kwargs):
        #number of neuron depends on number Outputs we want
        #each neuron's weights depends on number of inputs it recienve's  
        self.neurons:list[Neuron] = [Neuron(n_inp,activation_fn=activation_fn ,**kwargs) for _ in range(n_out)]

    def __call__(self, x):
       
        out:list[Tensor] = [n(x) for n in self.neurons]
        
        return out
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
    
    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"
    
class MLP(Module):
    def __init__(self, n_inp:int, Layer_outs:list[int],activation_fn: Literal["tanh", "relu"] = "tanh"):
        input_size = [n_inp] + Layer_outs
        self.layers:list[Layer] = [
            Layer(
                n_inp=input_size[out],
                n_out=input_size[out + 1],
                non_linear=(out != len(Layer_outs) - 1),
                activation_fn=activation_fn
            )
            for out in range(len(Layer_outs))
        ]

    def __call__(self, x):
        for layer in self.layers:
       
            x = layer(x)
       
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"


def GradientDescent(model,x,y,epoch=200):
    lostHistory=[]
    ypred:list = model(x) #gives a list of outputs 
    ypred:Tensor = Tensor.cat(ypred, axis=1) #list of Q Tensors(mx1) to be converted back to Tensor shape(mxQ) rows to be contatinate 
    if not isinstance(y,Tensor):
        y = Tensor(y,label='Output').transpose()
    if y.data.shape != ypred.data.shape:
        msg = (
            "prediction and true output shape mismatch: "
            f"y.shape={y.data.shape}, ypred.shape={ypred.data.shape}"
        )
        raise ValueError(msg)
    
    for _ in range(epoch):
        # print(np.shape(y.data))
        ypred = model(x) #gives a list of outputs 
        ypred = Tensor.cat(ypred, axis=1) #list to be converted back to Tensors
        loss = ((ypred - y) ** 2).mean(axis=1)

        for p in model.parameters():
            p.grad = 0.0

        loss.backward()

        for p in model.parameters():
            p.data -= 0.01 * p.grad
        lostHistory.append(loss.data)
        # print("loss:", loss.data[0])
    return lostHistory