# Fast and versatile Bayesian spike and slab
### Introduction
This work builds upon Naqvi et al. Fast Laplace Approximation for Sparse Bayesian Spike and Slab Models, by adding in a random effect to produce more robust posterior inclusion probabilities (PIPs).

When the main goal is variable selection, we treat the variance components as hyperparameters and estimate them by empirical Bayes (type II maximum likelihood).
The model implemented can be described as follows:

$$ 
\begin{aligned}
    &Y | \beta , \mu , \sigma^2 \sim \text{N} (X \beta + Z \mu, \sigma^2 ) \\
    & \beta | \tau_1^2 , \tau_0^2 \sim \pi \text{N}(0, \tau_1^2) + (1-\pi) \text{N}(0, \tau_0^2) \\
    & \mu | \delta^2 \sim \text{N} (0, \delta^2) \\
    & P(\beta | Y, \mu, \sigma_2, \tau_1^2 , \tau_0^2 , \delta^2 ) = \frac{
    P(Y | \beta , \mu , \sigma^2 ) P(\beta | \tau_1^2 , \tau_0^2 ) P (\mu | \delta^2 )
    }{
    P(Y | \mu, \sigma^2 , \tau_1^2 , \tau_0^2 ) P( \mu | \delta^2 )
    }
\end{aligned}
$$

Where $Y \in \mathbb{R}^{N\times 1}$ is a response vector; $X \in \mathbb{R}^{N \times p}$ is a matrix of predictors; $Z \in \mathbb{R}^{N \times m}$ is a one-
hot-encoded group ID matrix; $\mu$ denotes the random effect vector; $\beta$ the (sparse) regression weights; $\sigma^2$ the fixed effects variance; $\tau_0^2$, $\tau_1^2$ the spike and slab variances, respectively; $\pi$ the spike and slab mixture proportion; $\delta^2$ the random effects variance component.

### Marginalizing random effects
First, we integrate out the random effects, $\mu$:

$$\begin{aligned}
    P(Y, \beta) &= \int P(Y | \beta , \mu , \sigma^2 ) P(\beta | \tau_1^2 , \tau_0^2 ) P (\mu | \delta^2 ) d \mu \\
    & = P(\beta | \tau_1^2 , \tau_0^2 ) \int P(Y | \beta , \mu , \sigma^2 ) P (\mu | \delta^2 ) d \mu
\end{aligned}$$

This is a well-known integral (Gaussian convolution): 
$$= P(\beta | \tau_1^2 , \tau_0^2 ) \text{N} (Y | X \beta, \Sigma ) ; $$ 

$$\Sigma = \delta^2 Z Z^T + \sigma^2 I$$

### Maximum a posteriori estimation
We now have a joint distribution for $Y$ and $\beta$, which is proportional to the posterior conditioned on $Y$:

$$ P(\beta | \tau_1^2 , \tau_0^2 ) \text{N} (Y | X \beta, \Sigma ) \propto  \int P(\beta | Y, \mu, \sigma_2, \tau_1^2 , \tau_0^2 , \delta^2 ) d \mu$$

Taking the logarithm of the LHS we observe the following differentiable function:

$$
\begin{aligned}
    \log P(Y, \beta ) = & - \frac{N}{2} \log 2 \pi - \frac{1}{2} \log | \Sigma | - \frac{1}{2} (Y - X \beta)^T \Sigma^-1 (Y - X \beta) \\
    & + \sum_{j=1}^p \log \Big( \pi \text{N} (\beta_j | 0, \tau_1^2 ) + (1 - \pi) \text{N} (\beta_j | 0, \tau_0^2 ) \Big) 
\end{aligned}
$$

As in Naqvi et. al, we approximate the posterior using the Laplace method. To deal with random effects, we iterate between updating the variance components &mdash; $\sigma^2$, $\delta^2$ &mdash; and the coefficient vector $\beta$.  First, we set $\sigma^2$ and $\delta^2$ to a constant. We maximize the posterior with respect to $\beta$. To do this, we need the gradients of the posterior with respect to $\beta$: 

The gradient of the likelihood is:

$$ \nabla_\beta \log P(Y | \beta , \sigma^2 ) = X^T \Sigma^{-1} Y - X^T \Sigma^{-1} X \beta $$

The gradient of the prior is: 

$$\nabla_{\beta_j} \log P (\beta_j | \tau_1^2 , \tau_0^2 ) = - \beta_j \frac{
\frac{\pi}{\tau_1^2} \text{N} ( \beta_j | 0, \tau_1^2 ) + \frac{(1-\pi)}{\tau_0^2} \text{N} (\beta_j | 0, \tau_0^2 )
}{
\pi \text{N} ( \beta_j | 0, \tau_1^2 ) + (1 - \pi) \text{N} ( \beta_j | 0, \tau_0^2 ) 
}$$

Given our current guesses of the variance components, we find an initial MAP estimate $\hat{\beta}(\sigma^2, \delta^2 )$ using the R function \texttt{optim} with \texttt{method = "BFGS"}, supplying the gradient of the likelihood and prior shown above. 

### Hessian calculation
We compute the Hessian at $\hat{\beta}$, which serves as a rough initial approximation of the Fisher information. The Hessian of the likelihood component is: 

$$ \nabla_\beta^2 \log P(Y | \beta, \sigma^2 ) = - X^T \Sigma^{-1} X $$

The Hessian of the prior is:


$$\nabla_\beta^2 \log P(\beta | \tau_1^2 , \tau_0^2 )= \text{diag} \Big( \frac{
\nabla_\beta^2 P( \beta | \tau_1^2 , \tau_0^2 ) 
}{P ( \beta | \tau_1^2 , \tau_0^2 )}-\Big( \frac{
\nabla_\beta P( \beta) 
}{P (\beta ) } \Big)^2 
\Big)$$

Where we have: 

$$ \nabla_{\beta_j} P( \beta) = - \beta_j \Big( \frac{\pi}{\tau_1^2} \text{N} (\beta_j | 0, \tau_1^2 ) + \frac{(1 - \pi)}{\tau_0^2 } \text{N} (\beta_j | 0, \tau_0^2 ) \Big)$$

$$ \nabla_{\beta_j}^2 P (\beta | \tau_1^2 , \tau_0^2 ) = \pi \Big( \frac{\beta_j^2 }{\tau_1^4} - \frac{1}{\tau_1^2} \Big) \text{N} (\beta_j | 0, \tau_1^2 ) + (1 - \pi) \Big( \frac{\beta_j^2 }{\tau_0^4} - \frac{1}{\tau_0^2} \Big) \text{N} ( \beta_j | 0, \tau_0^2 ) $$

Going forward, we write the full Hessian as

$$\text{H}(\beta) = \nabla_\beta^2 \log P( Y | \beta, \sigma^2 ) + \nabla_\beta^2 \log P( \beta | \tau_1^2 , \tau_0^2 ) $$

and its matrix inverse as $\text{H}^{-1} (\beta) $.

### Estimation of the variance components
We use the Laplace approximation to obtain the marginal distribution, $P(Y | \sigma^2, \tau_1^2 , \tau_0^2 , \delta^2 )$ from which to estimate $\sigma^2$ and $\delta^2$ empirically. The Laplace approximation is found by Taylor expansion of the log exponent of the posterior as follows: 

$$ 
\begin{aligned}
    P (\beta, Y | \sigma^2 , \tau_1^2 , \tau_0^2 , \delta^2 ) &= \exp \Big ( \log P (\beta, Y | \sigma^2 , \tau_1^2 , \tau_0^2 , \delta^2 ) \Big) \\
    & \approx \exp \Big( \log P (\hat{\beta}, Y | \sigma^2 , \tau_1^2 , \tau_0^2 , \delta^2 ) \\
    & + (\beta - \hat{\beta})^T \text{H} (\hat{\beta}) (\beta - \hat{\beta} \Big) \\
    & = P(\hat{\beta}, Y | \sigma^2 , \tau_1^2, \tau_0^2 , \delta^2 ) \cdot \text{N} \big( \beta | \hat{\beta} , - \text{H}^{-1} (\hat{\beta}) \big)
\end{aligned}
 $$
 
As this is a Gaussian integral, we can derive a closed-form expression for the approximate marginal distribution:

$$
\begin{aligned}
    \int P (\beta, Y | \sigma^2 , \tau_1^2 , \tau_0^2 , \delta^2 ) d \beta &= P(\hat{\beta}, Y | \sigma^2 , \tau_1^2, \tau_0^2 , \delta^2 ) \cdot \int \text{N} \big( \beta | \hat{\beta} , - \text{H}^{-1} (\hat{\beta}) \big) d \beta  \\
    &= P(\hat{\beta}, Y | \sigma^2 , \tau_1^2, \tau_0^2 , \delta^2 ) (2 \pi)^{p/2} \text{det} (- H (\hat{\beta} ) )^{-1/2}
\end{aligned}
$$

The log marginal is then: 

$$ 
\log P(Y | \sigma^2 , \tau_1^2 , \tau_0^2 , \delta^2 ) = - \frac{1}{2} \log | \Sigma | - \frac{1}{2} (Y - X \beta )^T \Sigma^{-1} (Y - X \beta)- \frac{1}{2} \log | - \text{H} ( \hat{\beta} ) |
$$

To obtain point estimates of $\sigma^2$ and $\delta^2$ by gradient methods, we first transform the parameter space to the real line using a log transform. We let $\theta = ( \log \sigma^2, \log \delta^2)$ (identically, $ (\sigma^2, \delta^2) = (e^{\theta_1}, e^{\theta_2})$). This enables us to perform unconstrained optimization over the real number plane. Note that we do not need to apply change of variables (scaling by the Jacobian determinant) because $\sigma^2$ and $\delta^2$ are hyperparameters for this specific use case. \\ \\
The gradient of the evidence is computed using the following matrix derivative identities: 

$$ \frac{\partial }{\partial \theta_i } \log | \Sigma | = \text{tr} (\Sigma^{-1} \partial_{\theta_i} \Sigma ) $$

$$ \frac{\partial}{\partial \theta_i} (Y - X \beta )^T \Sigma^{-1} (Y - X \beta ) = - (Y - X \beta)^T \Sigma^{-1} ( \partial_{\theta_i} \Sigma ) \Sigma^{-1} (Y - X \beta )$$

$$ \frac{\partial}{\partial \theta} \log |- \text{H} \big( \beta) | = \text{tr} ( - \text{H} ^{-1} ( \beta) ( - \partial_\theta \text{H} (\beta) ) \big) $$

$$ \partial_{\theta_i} (- \text{H} (\beta) ) = - X^T \Sigma^{-1} (\partial_{\theta_i} \Sigma ) \Sigma^{-1} X $$

So we have: 

$$ 
\begin{aligned}
    \nabla_{\theta_i} \log P(Y | \sigma^2 , \tau_1^2 , \tau_0^2 , \delta^2 ) = & - \frac{1}{2} \text{tr} (\Sigma^{-1} \partial_{\theta_i} \Sigma ) + \frac{1}{2} (Y - X \beta) ^T \Sigma^{-1} (\partial_{\theta_i} \Sigma ) \Sigma^{-1} (Y - X \beta ) \\
    &- \frac{1}{2} \text{tr} (- \text{H} ^{-1} (\beta ) ) ( - \partial_{\theta_i} \text{H} ( \beta ) ) 
\end{aligned}
$$

### Iterative convergence
We repeat the previous steps (MAP estimation, Hessian computation, evidence maximization) until  the following convergence criterion is met: 

$$ \log P( \beta^{(t)} | X, Y, \Theta^{t} ) - \log P (\beta^{(t-1)} | X, Y \Theta ^{(t-1)} ) \leq 0.0001 $$ 

where $\beta^{(t)}$ and $\Theta^{(t)} $ denote the current parameter estimates for the beta coefficients and the variance components ($\Theta$ for shorthand notation) at the $t$-th iteration. 

### Posterior inclusion probabilities

Once we have estimated the MAP for $\beta$, $\sigma^2$ and $\delta^2$, the PIPs can be obtained as: 

$$ \text{PIP}(\beta_j) = \int \frac{
\pi \text{N} (\beta_j | 0, \tau_1^2 ) 
}{
\pi \text{N} (\beta_j | 0, \tau_1^2 ) + (1 - \pi ) \text{N} (\beta_j | 0, \tau_0^2 ) 
} \cdot \text{N} (\beta_j | \hat{\beta}_j - \text{H}^{-1}(\hat{\beta})_{jj} d \beta_j $$

While this integral has no closed form, it is a simple 1-dimensional integral that can be efficiently computed using numerical methods. 
