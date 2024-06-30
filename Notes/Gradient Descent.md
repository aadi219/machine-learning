Technique used in many machine learning models in order to find the *local minima* of a *differentiable cost function* w.r.t the parameters being used in the model.
### Process
1. Start with some initial value of your parameter $\theta$.
2. Keep changing $\theta$ little by little to reduce $cost(\theta)$. *How?*
3. Do it until the algorithm converges - minimum found.

Gradient Descent asks that for any point in the plane of the cost function w.r.t the parameters, in what direction should you move the point in your next iteration *so as to move downhill as quickly as possible*.
Gradient descent may converge at *different local minima* depending on the initial values of the parameters chosen or the values of other hyperparameters such as *learning rate*.

The values of the parameters are changed as follows:
$$\theta_i:=\theta_i - \alpha\cdot\frac{\partial{cost(\theta_i)}}{\partial {\theta_i}}$$
*The derivative of a function defines the direction of steepest **ascent***. *Therefore the derivative is subtracted in order to move in the direction of descent.*
$$\frac{\partial cost(\theta_i)}{\partial theta_i}= \frac{\partial}{\partial \theta_i}(h(x)-y)^2 = (h(x) - y) \cdot x_i$$
	for one training example
When taking all examples into account, the formula becomes
$$\theta_i := \theta_i - \alpha\cdot \sum_{j=0}^{m}(h(x_j) - y_j) \cdot x_{ij}$$

### Batch Gradient Descent
Refers to using the **entire training set** in **each iteration** of the learning process.
If the dataset is extremely large then the computation of the sum of costs for every single parameter at every single iteration is extremely slow and computationally intensive.

### Stochastic Gradient Descent
SGD involves iterating over all the training examples and computing the change in the parameters at each iteration with respective to a single example.
