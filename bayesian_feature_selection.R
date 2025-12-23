library(MatrixGenerics)
set.seed(1)

setwd('/path/to/data')
X = readRDS('3.prep_data___beta_sign___kidney.rds')
Y = X$dose_vec
Z = X[, 1, drop = FALSE]
Z = model.matrix(~ cc_vec - 1, data = as.data.frame(Z))
X = X[, 4:ncol(X)]

# log likelihood 
L = function(x, S, S_inv){
  B = x
  term1 = -1/2 * log(det(S))
  term2 = -1/2 * t(Y - X %*% B) %*% S_inv %*% (Y - X %*% B)
  return(term1 + term2)
}

# gradient of log likelihood
gradLikelihood = function(x, S, S_inv){
  B = x
  p = length(x)/2
  term1 = t(X) %*% S_inv %*% Y - t(X) %*% S_inv %*% X %*% B
  return(term1)
}

# hessian of log likelihood
hessLikelihood = function(x, S, S_inv){
  B = x
  term1 = - t(X) %*% S_inv %*% X
  return(term1)
}

# log prior
prior = function(x){
  B = x
  sum(log(p * dnorm(B, 0, sqrt(tau1)) + (1-p) * dnorm(B, 0, sqrt(tau0))))
}

# gradient of log prior
gradPrior = function(x){
  B = x
  N1 = dnorm(B, 0, sqrt(tau1))
  N0 = dnorm(B, 0, sqrt(tau0))
  
  term1 = p * -B / tau1 * N1 + (1-p) * -B / tau0 * N0
  term2 = p * N1 + (1-p) * N0
  
  return(term1 / term2)
}

# hessian of log prior
hessPrior = function(x){
  B = x
  
  N1 = dnorm(B, 0, sqrt(tau1))
  N0 = dnorm(B, 0, sqrt(tau0))
  
  nabla_prior = gradPrior(x)
  
  term1 = p * (B^2/tau1^2 - 1/tau1) * N1 + (1-p) * (B^2 / tau0^2 - 1/tau0) * N0
  term2 = p * N1 + (1-p) * N0
  hessU = term1/term2 - nabla_prior^2 
  
  return(diag(as.vector(hessU)))
}

# full gradient
Grad = function(x, S, S_inv){
  -1 * (gradLikelihood(x, S, S_inv) + gradPrior(x))
}

# full hessian
Hess = function(x, S, S_inv){
  hessLikelihood(x, S, S_inv) + hessPrior(x)
}

# log posterior
objectiveFunction = function(x, S, S_inv){
  - L(x, S, S_inv) - prior(x)
}

# log marginal posterior (for laplace)
objectiveFunction2 = function(x, map){
  
  B_map = as.numeric(map)
  theta1_ = x[1]
  theta2_ = x[2]
  
  I = diag(nrow(X))
  S = exp(theta2_) * Z %*% t(Z) + exp(theta1_) * I
  S_inv = solve(S)
  
  H = Hess(map, S, S_inv)
  H_inv = - solve(H + 1e-6 * diag(ncol(H)))
  
  term1 = -1/2 * log(det(S))
  term2 = -1/2 * t(Y - X %*% B_map) %*% S_inv %*% (Y - X %*% B_map)
  
  Hchol = chol(-H)
  term3 = - sum(log(diag(Hchol)))
  
  term1 = as.numeric(term1) ; term2 = as.numeric(term2) ; term3 = as.numeric(term3)
  return(- term1 - term2 - term3)
}

# gradient of log marginal posterior
tr = function(x){ sum(diag(x))}

Grad2 = function(x, map){
  B_map = as.numeric(map)
  theta1_ = x[1]
  theta2_ = x[2]
  
  I = diag(nrow(X))
  S = exp(theta2_) * Z %*% t(Z) + exp(theta1_) * I
  S_inv = solve(S)
  
  H = Hess(map, S, S_inv)
  H_inv = - solve(H + 1e-6 * diag(ncol(H)))
  
  deltaS1 = exp(theta1_) * I
  deltaS2 = exp(theta2_) * ZZt
  
  deltaH1 = - t(X) %*% S_inv %*% deltaS1 %*% S_inv %*% X
  deltaH2 = - t(X) %*% S_inv %*% deltaS2 %*% S_inv %*% X
  
  term1 = -1/2 * tr(S_inv %*% deltaS1)
  term2 = 1/2 * t(Y - X %*% B_map) %*% S_inv %*% deltaS1 %*% S_inv %*% (Y - X %*% B_map)
  term3 = -1/2 * tr(-H_inv %*% - deltaH1)
  
  term4 = -1/2 * tr(S_inv %*% deltaS2)
  term5 = 1/2 * t(Y - X %*% B_map) %*% S_inv %*% deltaS2 %*% S_inv %*% (Y - X %*% B_map)
  term6 = -1/2 * tr(-H_inv %*% - deltaH2)
  
  partial_theta1 = term1 + term2 + term3
  partial_theta2 = term4 + term5 + term6
  
  return(c(-partial_theta1, -partial_theta2))
}

# posterior inclusion probabilities
# B
pZ = function(B, posterior_mu, posterior_var){
  term1 = p * dnorm(B, 0, sqrt(tau1))
  term2 = (1-p) * dnorm(B, 0, sqrt(tau0))
  term3 = dnorm(B, posterior_mu, sqrt(posterior_var))
  return(term1 / (term1 + term2) * term3)
}

master <- NULL

B = rep(0, ncol(X))

tau1 = 0.01
tau0 = 0.0001

p = 0.1
eta = 0.01

X = as.matrix(X)
X = scale(X)
Y = scale(Y)

XtX = t(X) %*% X
XtY = t(X) %*% Y 
ZZt = Z %*% t(Z)

theta1 = log(0.5)
theta2 = log(0.5)



message('initializing model with OLS estimate')
B = solve(t(X) %*% X + 1e-3 * diag(ncol(X)), t(X) %*% Y)
map = B

loss0 <- Inf
loss1 <- 0
counter <- 0
tol <- 1e-6
max_iter <- 10

while (loss0 - loss1 >= 1e-6 && counter < max_iter){
  I = diag(nrow(X))
  S = exp(theta2) * Z %*% t(Z) + exp(theta1) * I
  S_inv = solve(S)
  
  loss0 = loss1
  loss1 <- objectiveFunction(as.numeric(map), S = S, S_inv = S_inv) %>% as.numeric()
  
  message('computing map estimates')
  bfgs = optim(par = as.numeric(map), fn = objectiveFunction, S = S, S_inv = S_inv, gr = Grad, method = 'L-BFGS-B',
               control = list(trace = 1, maxit = 1000))
  
  map = bfgs$par
  
  
  message('performing laplace approximation')
  H = Hess(map, S, S_inv)
  Sigma = - solve(H + 1e-6 * diag(ncol(H)))
  s = diag(Sigma)
  
  message('empirical point estimation of variance componenets')
  bfgs = optim(par = c(theta1, theta2), fn = objectiveFunction2, map = map,
               gr = Grad2, method = 'L-BFGS-B', 
               control = list(trace = 1, maxit = 100), lower = c(-7,-7))
  
  theta1 = bfgs$par[1]
  theta2 = bfgs$par[2]
  theta1 ; theta2
  
  counter = counter + 1
  message('iter: ', counter, ' | loss: ', loss1)
}

# one last update
I = diag(nrow(X))
S = exp(theta2) * Z %*% t(Z) + exp(theta1) * I
S_inv = solve(S)
bfgs = optim(par = as.numeric(map), fn = objectiveFunction, S = S, S_inv = S_inv, gr = Grad, method = 'L-BFGS-B',
             control = list(trace = 1, maxit = 1000))
map = bfgs$par
H = Hess(map, S, S_inv)
Sigma = - solve(H + 1e-6 * diag(ncol(H)))
s = diag(Sigma)


B_map = map

B_var = s

# PIPs
message('computing posterior inclusion probabilities')
pz = numeric()
for (i in 1:length(B_map)){
  b_mu = B_map[i]
  b_var = B_var[i]
  
  pz[i] = integrate(f = pZ, posterior_mu = b_mu, posterior_var = b_var,
                    lower = b_mu - 6 * sqrt(b_var),
                    upper = b_mu + 6 * sqrt(b_var))$value
}
plot(pz)
plot(B_map)


# temporary. out of time to produce results
plot(B_init)
plot(B_map)
ix = which(abs(B_map) >= 0.05)
colnames(X)[ix]

# -------------------------------------------------------------------------


# CIs
message('computing marginal credible intervals')
ci0 = ci1 = ci_u = ci_d = list()
n = nrow(Sigma)/2
for (i in 1:length(U_map)){
  u_mu = U_map[i]
  d_mu = D_map[i]
  
  u_var = U_var[i]
  d_var = D_var[i]
  
  u_d_covar = Sigma[i, i + n]
  
  mu1 = u_mu + 1/2 * d_mu
  mu0 = u_mu - 1/2 * d_mu
  
  v1 = u_var + 1/4 * d_var + u_d_covar
  v0 = u_var + 1/4 * d_var - u_d_covar
  
  ci_u[[i]] = qnorm(c(0.025, 0.975), mean = u_mu, sd = sqrt(u_var))
  ci_d[[i]] = qnorm(c(0.025, 0.975), mean = d_mu, sd = sqrt(d_var))
  
  ci0[[i]] = qnorm(c(0.025, 0.975), mean = mu0, sd = sqrt(v0))
  ci1[[i]] = qnorm(c(0.025, 0.975), mean = mu1, sd = sqrt(v1))
}

ci_u = do.call(what = rbind, args = ci_u)
colnames(ci_u) = c('U_lower', 'U_upper')

ci_d = do.call(what = rbind, args = ci_d)
colnames(ci_d) = c('D_lower', 'D_upper')

ci0 = do.call(what = rbind, args = ci0)
colnames(ci0) = c('B0_lower', 'B0_upper')

ci1 = do.call(what = rbind, args = ci1)
colnames(ci1) = c('B1_lower', 'B1_upper')

message('storing results')
convergence = bfgs$convergence
id = gene_pairs_i$id[1]

df = cbind(pip_u = pz, pip_d = pw, U_map, ci_u, D_map, ci_d, B0_map, ci0, B1_map, ci1, convergence)
df = as.data.frame(df)
df = cbind(id, features, df)

if (iter == 1){
  master = df
} else {
  master = rbind(master, df)
}

if (iter %% 100 == 0){
  setwd(apoe_ccre_objects)
  saveRDS(master, paste0('4.spike_slab___', chr, '.rds'))
}



setwd(apoe_ccre_objects)

saveRDS(master, paste0('4.spike_slab___', chr, '.rds'))
