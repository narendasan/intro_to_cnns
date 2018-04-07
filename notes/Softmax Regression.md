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
2. What if we assigned how important each pixel is for each possible label?
   1. We can encode this as a matrix of parameters $\boldsymbol w$ size (HxW)x10 
3. We can then given the "optimal parameters" $\boldsymbol w^*$ calculate which label best matches an image using $y^{'} = {\boldsymbol {w}^*}^Tx^{(i)}$
4. There might also be some labels that occur more often than others so we can add a bias $b$ for each class encoded as a vector size 10, $y^{'} = {\boldsymbol {w}^*}^Tx^{(i)} + b$

How do we figure out the best values for $w^*$?

Simple Solution is linear regression: not vary accurate for multiclass classification

What if we tried a random set of parameters, saw how well it did and then if its good keep doing that otherwise change?

This is called Gradient Decent (Kind of like fancy Newtons Method) - Follow the slope (Gradient) of the space $\nabla _w \mathcal L$ (i.e. $\frac{\partial \mathcal L}{\partial w}$) made by error (loss) $\mathcal L$ caused by the parameters until the error does not get any better

$\boldsymbol w_{t+1} = \boldsymbol w_t - \alpha \nabla_{\boldsymbol w_t}$

 	1. How do we get the error $\mathcal L$?
      	1. Up to you! -> Linear regression uses something called Least Squares 
           	1. $\mathcal L = \sum\limits_{(x^{(i)}, y^{(i)}) \in \mathbb D} \frac{1}{2}|y^{(i)} -  ({\boldsymbol{w}^*}^Tx^{(i)} + b) |_2^2$
           	2. so we want to $\min\limits_w \sum\limits_{(x^{(i)}, y^{(i)}) \in \mathbb D} \frac{1}{2}|y^{(i)} -  ({\boldsymbol{w}^*}^Tx^{(i)} + b) |_2^2$

##  Softmax Regression:

What if we want to get the probability that a image is any particular class?



Can you solve this? 

Using Gradient decent -> 
$$
\nabla_{\boldsymbol w} = \sum\limits_{(x^{(i)}, y^{(i)}) \in \mathbb D} \frac{-y^{(i)}\phi(x)exp(-y^{(i)} \boldsymbol w^T\phi(x^{(i)})}{1 + exp(-y^{(i)}\boldsymbol w^T\phi(x^{(i)}  )}
$$
So what we do is we calculate the gradient, apply a scalar called a stepsize $\alpha$ to each parameter to move it closer toward the local minimum (i.e. move closer to the optimal values)













$$
\min\limits_w \sum\limits_{(x^{(i)}, y^{(i)}) \in \mathbb D} log(1 + exp(-y^{(i)}\boldsymbol{w}^T\phi(x^{(i)})))\\
\text{Where } \boldsymbol w \text{ are the parameters (weights) of the model}\\
\text{and } \phi(x^{(i)}) \text{ is what is a called a feature transform}
$$
