class Linear():
    r"""Make linear(which does matrix multiplication) model with given parameters.

    Parameters
    ----------
    w : {torch.tensor}, default = None
        Gets weight parameter.  In forward process, no reshaping happens. So becareful to render adequate size of
        tensor
    b : {torch.tensor}, default = None
        Gets bias parameter, only applied to calculation when is_bias = True
    is_bias : {bool}, default = True
        If False, doesn't include bias in linear calculation.

    Attributes
    ----------
    g : tensor
        gradient for each parameter.

    Note
    ----------
    g(gradient) attribute for each paramter is not from PyTorch's automatic differentiation engine.

    Examples
    ----------
    x = torch.randn(1000, 10)
    y = torch.randn(1000)
    linear_model = Linear(torch.randn(10, 1), is_bias= False)
    model_output = linear_model(x)
    mean_square_error = (model_output.squeeze(-1) - y).unsqueeze(-1).pow(2).sqrt()
    model_output.backward()
    weight_grad = model_output.w.g
    """
    def __init__(self, w, b, is_bias = True):
        self.w, self.b = w, b

    def forward(self, x):
        """Calculate linear multiplication and return the result
        """
        self.inp = x
        if is_bias:
            self.out = self.inp@self.w + self.b
        else:
            self.out = self.inp@self.w
        return self.out

    def backward(self):
        """Caculate derivative of each parameter including input:x, as they can be used for previous layer.

        Note
        ------------
        As for now, we used python's matrix multiplication operator @(at).
        However, we recommend you to use broadcasting or einsum for faster and more memory-efficient calculation.

        """
        self.inp.g = self.out.g @ self.w.t()
        self.w.g = (self.inp.unsqueeze(-1) * self.out.g.unsqueeze(1)).sum(0)
        self.b.g = self.out.g.sum(0)
