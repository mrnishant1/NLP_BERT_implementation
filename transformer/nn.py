from typing import Literal
import random
import warnings
from .tensor import Tensor
import numpy as np


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self,n_inp,non_linear=True,activation_fn: Literal["tanh", "relu"] = "relu"):
        std = np.sqrt(2 / n_inp)       
        self.W = Tensor(np.random.randn(1, n_inp) * std,label="w")

        self.b = Tensor([[0.0]],label="bias")

        self.non_linear = non_linear
        self.activation_fn = activation_fn

    def __call__(self, x):

        # x is expected to be:
        # (batch_size, n_inp)

        x = Tensor(np.asarray(x, dtype=float)) if not isinstance(x, Tensor) else x

        # W: (1, n_inp)
        # x.T: (n_inp, batch_size)
        #
        # W @ x.T -> (1, batch_size) 
        act = (self.W * x.transpose()) + self.b

        # Return shape:
        # (batch_size, 1)
        act = act.transpose()

        # Final layer: return raw logits
        if not self.non_linear:
            out = act
            return out

        # Hidden layer activation
        activation = getattr(act, self.activation_fn)
        return activation()

    def parameters(self):
        return [self.W, self.b]

    def __repr__(self):
        activation = self.activation_fn if self.non_linear else "linear"
        return f"{activation} Neuron({self.W.data.shape[1]})"


class Layer(Module):
    def __init__(self,layer_id,n_inp,n_out,activation_fn="relu",non_linear=True):
        self.layer_id = layer_id

        self.neurons = [Neuron(n_inp=n_inp,activation_fn=activation_fn,non_linear=non_linear) for _ in range(n_out)]

    def __call__(self, x):

        # Each neuron produces:
        #
        # (batch_size, 1)
        #
        # Combine them into:
        #
        # Tensor -> (batch_size, n_out)
        outputs = [neuron(x) for neuron in self.neurons]
        outputs = Tensor.cat(outputs, axis=1)
        return outputs

    def parameters(self):
        return [ p for neuron in self.neurons for p in neuron.parameters()]

    def __repr__(self):
        return (f"Layer of " f"[{', '.join(str(n) for n in self.neurons)}]")


class MLP(Module):

    def __init__( self, n_inp: int, Layer_outs: list[int], activation_fn="relu"):
        input_size = [n_inp] + Layer_outs
        self.layers = [
            Layer(
                layer_id=i,
                n_inp=input_size[i],
                n_out=input_size[i + 1],

                # Hidden layers -> activation
                # Final layer -> linear
                non_linear=(i != len(Layer_outs) - 1),
                

                activation_fn=activation_fn
            )
            for i in range(len(Layer_outs))
        ]

    def predict(self, x):

        # Convert input to Tensor once
        if not isinstance(x, Tensor):
            x = Tensor(np.asarray(x, dtype=float))

        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [ p for layer in self.layers for p in layer.parameters()]

    def Fit(
        self,
        x,
        y,
        epoch=200,
        lr=0.0001,
        batch_size=32,
        loss_type: Literal["mse", "cross_entropy"] = "mse"
    ):
        n = len(y)
        epoch_loss_history = []

        for _ in range(epoch):

            # Shuffle every epoch
            indices = list(range(n))
            random.shuffle(indices)

            epoch_loss = 0.0
            num_batches = 0

            for start in range(0, n, batch_size):

                batch_indices = indices[start:start + batch_size]

                x_batch = [x[i] for i in batch_indices]
                y_batch = [y[i] for i in batch_indices]

                # Forward pass
                ypred = self.predict(x_batch)

                # Convert target to Tensor
                y_tensor = (
                    y_batch
                    if isinstance(y_batch, Tensor)
                    else Tensor(np.asarray(y_batch, dtype=float), label="Output")
                )

                # Check shape
                if y_tensor.data.shape != ypred.data.shape:
                    raise ValueError(
                        "Prediction and target shape mismatch: "
                        f"y.shape={y_tensor.data.shape}, "
                        f"ypred.shape={ypred.data.shape}"
                    )

                # Forward + backward + update
                loss = self._GradientDescent(
                    y=y_tensor,
                    ypred=ypred,
                    lr=lr,
                    loss_type=loss_type
                )
                # Store scalar loss
                epoch_loss += float(np.asarray(loss.data).mean())
                num_batches += 1

            # Average batch losses
            epoch_loss_history.append(epoch_loss / num_batches)

        return epoch_loss_history


    def _GradientDescent(self, y, ypred, loss_type, lr=0.1):

        # Convert y to Tensor
        if not isinstance(y, Tensor):
            y = Tensor(
                np.asarray(y, dtype=float),
                label="Output"
            )

        # Reset gradients
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

        # Calculate loss
        loss = self._Calculate_Loss(
            ypred=ypred,
            y=y,
            loss_type=loss_type
        )

        # Backprop
        loss.backward()

        # Gradient descent
        for p in self.parameters():
            p.data -= lr * p.grad

        return loss


    def _Calculate_Loss(self, ypred: Tensor, y: Tensor, loss_type):

        if loss_type == "mse":

            # Element-wise squared error
            squared_error = (ypred - y) ** 2
            # Mean over all elements
            return squared_error.mean()

        elif loss_type == "cross_entropy":

            warnings.warn(
                "Using binary cross entropy."
            )

            # Ignore labels == -100
            mask = Tensor(
                (np.asarray(y.data) != -100).astype(float)
            )

            # Replace ignored labels with 0
            target = Tensor(
                np.where(
                    np.asarray(y.data) == -100,
                    0,
                    np.asarray(y.data)
                )
            )

            eps = 1e-7

            # IMPORTANT:
            # Don't do `ypred += eps`
            # because that modifies the prediction Tensor in-place.
            safe_ypred = ypred + eps

            # BCE:
            #
            # -[y log(p) + (1-y) log(1-p)]
            #
            positive = target.elementwise_mul(safe_ypred.log())

            negative = (
                Tensor(1.0) - target
            ).elementwise_mul(
                (Tensor(1.0) - safe_ypred).log()
            )

            loss = (positive + negative).elementwise_mul(mask)

            # Average only valid elements
            number_of_valid = mask.sum()

            return loss.elementwise_mul(
                number_of_valid ** -1
            ).sum()

        else:
            raise ValueError(
                f"Unknown loss type: {loss_type}"
            )
            
    def __repr__(self):
        return (
            f"MLP of "
            f"[{', '.join(str(layer) for layer in self.layers)}]"
        )
        
