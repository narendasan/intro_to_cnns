$$
\text{Let } \mathbb D = \{(x^{(i)}, y^{(i)}\}_{i=1}^N\\
\text{where } x^{(i)} \text{ is an image} \\
y^{(i)} \text{ is a label}
$$

How can we design a function that accurately maps 
$$
f:X \rightarrow Y
$$
i.e map an image to a label using a function.

1. How are images encoded (assume grayscale) ? 
   1. Pixel values from 0-255
2. What if we assigned how important each pixel is 

What if we assigned 

##  Softmax Regression:



Can you solve this? 

Using Gradient decent -> 
$$
\nabla_{\boldsymbol w} = \sum\limits_{(x^{(i)}, y^{(i)}) \in \mathbb D} \frac{-y^{(i)}\phi(x)exp(-y^{(i)} \boldsymbol w^T\phi(x^{(i)})}{1 + exp(-y^{(i)}\boldsymbol w^T\phi(x^{(i)}  )}
$$
So what we do is we calculate the gradient, apply a scalar called a stepsize $\alpha$ to each parameter to move it closer toward the local minimum (i.e. move closer to the optimal values)
$$
\boldsymbol w_{t+1} = \boldsymbol w_t - \alpha \nabla_{\boldsymbol w_t}
$$







$$
\min\limits_w \sum\limits_{(x^{(i)}, y^{(i)}) \in \mathbb D} log(1 + exp(-y^{(i)}\boldsymbol{w}^T\phi(x^{(i)})))\\
\text{Where } \boldsymbol w \text{ are the parameters (weights) of the model}\\
\text{and } \phi(x^{(i)}) \text{ is what is a called a feature transform}
$$
