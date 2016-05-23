data {
  int<lower=1> N;                        // size of field data
  int<lower=1> M;                        // size of computer model output
  int<lower=1> D;                        // x-input dimension
  int<lower=1> L;                        // t-input dimension
  vector[D] x1[N];                       // x-input values where the field observation                                          // are available
  vector[D] x2[M];                       // x-input values where computer model is                                             // evaluated
  vector[L] t[M];                        // t-input values where computer model is                                             // evaluated
  vector[N] y;                           // field data
  vector[M] eta;                         // computer model output
}

transformed data {
  vector[N+M] z;                         // field data and computer model output                                               // combined
  z <- append_row(y, eta);
}

parameters {
  real mu;                               // constant mean of the computer model GP
  real<lower=0> lambda_inv;              // variance of the computer-model GP
  real<lower=0> lambda_inv_d;            // variance of the discrepancy term GP
  real<lower=0> beta_x[D];               // correlation parameters for x
  real<lower=0> beta_t[L];               // correlation parameters for t (theta)
  real<lower=0> beta_d[D];               // correlation parameters in the discrepancy                                          // GP moldel (for x)
  real<lower=0> sigma_sq;                // observation error variance
  vector<lower=5, upper=15>[L] theta;    // unknown computer model input to be                                                 // estimated
}
model {
  real c1[D];
  real c2[L];
  real c3[D];
  matrix[N,N] Sigma1;                    // first diagonal block of the covariance                                             // matrix corresponding to field observations
  matrix[M,M] Sigma2;                    // second diagonal block of the covariance                                            // matrix corresponding to computer model                                             // output
  matrix[N,M] Sigma12;                   // off-diagonal block of the covariance                                               // matrix
  matrix[N+M,N+M] Sigma;                 // covariance matrix of the combined data
  matrix[N+M,N+M] CH;                    // cholesky decomposition of the covariance                                           // matrix
  for (i in 1:(N-1)) {
    Sigma1[i,i] <- lambda_inv + lambda_inv_d + sigma_sq;
    for (j in (i+1):N) {
      for (d in 1:D) c1[d] <- - beta_x[d] * pow(x1[i,d] - x1[j,d], 2);
      for (d in 1:D) c3[d] <- - beta_d[d] * pow(x1[i,d] - x1[j,d], 2);
      Sigma1[i,j] <- lambda_inv * exp(sum(c1)) + lambda_inv_d * exp(sum(c3));
      Sigma1[j,i] <- Sigma1[i,j];
    }
  }
  Sigma1[N,N] <- lambda_inv + lambda_inv_d + sigma_sq;

  for (i in 1:(M-1)) {
    Sigma2[i,i] <- lambda_inv;
    for (j in (i+1):M) {
      for (d in 1:D) c1[d] <- - beta_x[d] * pow(x2[i,d] - x2[j,d], 2);
      for (l in 1:L) c2[l] <- - beta_t[l] * pow(t[i,l] - t[j,l], 2);
      Sigma2[i,j] <- lambda_inv * exp(sum(c1) + sum(c2)) ;
      Sigma2[j,i] <- Sigma2[i,j];
    }
  }
  Sigma2[M,M] <- lambda_inv;

  for (i in 1:N) {
    for (j in 1:M) {
      for (d in 1:D) c1[d] <- - beta_x[d] * pow(x1[i,d] - x2[j,d], 2);
      for (l in 1:L) c2[l] <- - beta_t[l] * pow(theta[l] - t[j,l], 2);
      Sigma12[i,j] <- lambda_inv * exp(sum(c1) + sum(c2));
    }
  }
  Sigma <- append_row(append_col(Sigma1, Sigma12), append_col(Sigma12', Sigma2));
  CH <- cholesky_decompose(Sigma);
  //priors:
  lambda_inv ~ inv_gamma(1,1);
  lambda_inv_d ~ inv_gamma(1,1);
  sigma_sq ~ inv_gamma(1,1);
  beta_x ~ gamma(10,1);
  beta_t ~ gamma(10,1);
  beta_d ~ gamma(10,1);
  mu ~ normal(0,10);
  theta ~ normal(10,5);
  // likelihood of the combined data:
  z ~ multi_normal_cholesky(rep_vector(mu, N + M), CH);
}
// generated quantities {
//   // Here we do the simulations from the posterior predictive distribution
//   vector[N_pred] y_rep ; # vector of same length as the data z
//   vector[N_pred] mu_pred ; # vector of same length as the data z
//   real c1[D];
//   real c2[L];
//   real c3[D];
//   matrix[N,N] Sigma1;                    // first diagonal block of the covariance matrix corresponding to field                                                  observations
//   matrix[M,M] Sigma2;                    // second diagonal block of the covariance matrix corresponding to computer                                              model output
//   matrix[N,M] Sigma12;                   // off-diagonal block of the covariance matrix
//   matrix[N+M,N+M] Sigma;                 // covariance matrix of the combined data
//   matrix[N+M,N+M] Sigma_inv;
//   matrix[N_pred,N_pred] Sigma_pred0;
//   matrix[N_pred,N_pred] Sigma_pred;
//   matrix[N_pred,N] Sigma_p1;
//   matrix[N_pred,M] Sigma_p2;
//   matrix[N_pred,N+M] Sigma_p;
//   for (i in 1:(N-1)) {
//     Sigma1[i,i] <- lambda_inv + lambda_inv_d + sigma_sq;
//     for (j in (i+1):N) {
//       for (d in 1:D) c1[d] <- - beta_x[d] * pow(x1[i,d] - x1[j,d], 2);
//       for (d in 1:D) c3[d] <- - beta_d[d] * pow(x1[i,d] - x1[j,d], 2);
//       Sigma1[i,j] <- lambda_inv * exp(sum(c1)) + lambda_inv_d * exp(sum(c3));
//       Sigma1[j,i] <- Sigma1[i,j];
//     }
//   }
//   Sigma1[N,N] <- lambda_inv + lambda_inv_d + sigma_sq;
// 
//   for (i in 1:(M-1)) {
//     Sigma2[i,i] <- lambda_inv;
//     for (j in (i+1):M) {
//       for (d in 1:D) c1[d] <- - beta_x[d] * pow(x2[i,d] - x2[j,d], 2);
//       for (l in 1:L) c2[l] <- - beta_t[l] * pow(t[i,l] - t[j,l], 2);
//       Sigma2[i,j] <- lambda_inv * exp(sum(c1) + sum(c2)) ;
//       Sigma2[j,i] <- Sigma2[i,j];
//     }
//   }
//   Sigma2[M,M] <- lambda_inv;
// 
//   for (i in 1:N) {
//     for (j in 1:M) {
//       for (d in 1:D) c1[d] <- - beta_x[d] * pow(x1[i,d] - x2[j,d], 2);
//       for (l in 1:L) c2[l] <- - beta_t[l] * pow(theta[l] - t[j,l], 2);
//       Sigma12[i,j] <- lambda_inv * exp(sum(c1) + sum(c2));
//     }
//   }
//   Sigma <- append_row(append_col(Sigma1, Sigma12), append_col(Sigma12', Sigma2));
//   Sigma_inv <- inverse(Sigma);
// 
//   for (i in 1:(N_pred-1)) {
//     Sigma_pred0[i,i] <- lambda_inv + lambda_inv_d + sigma_sq;
//     for (j in (i+1):N_pred) {
//       for (d in 1:D) c1[d] <- - beta_x[d] * pow(x_pred[i,d] - x_pred[j,d], 2);
//       for (d in 1:D) c3[d] <- - beta_d[d] * pow(x_pred[i,d] - x_pred[j,d], 2);
//       Sigma_pred0[i,j] <- lambda_inv * exp(sum(c1)) + lambda_inv_d * exp(sum(c3));
//       Sigma_pred0[j,i] <- Sigma_pred0[i,j];
//     }
//   }
//   Sigma_pred0[N_pred,N_pred] <- lambda_inv + lambda_inv_d + sigma_sq;
//   
//   for (i in 1:N_pred) {
//     for (j in 1:N) {
//       for (d in 1:D) c1[d] <- - beta_x[d] * pow(x_pred[i,d] - x1[j,d], 2);
//       for (d in 1:D) c3[d] <- - beta_d[d] * pow(x_pred[i,d] - x1[j,d], 2);
//       Sigma_p1[i,j] <- lambda_inv * exp(sum(c1)) + lambda_inv_d * exp(sum(c3));
//     }
//   }
//   for (i in 1:N_pred) {
//     for (j in 1:M) {
//       for (d in 1:D) c1[d] <- - beta_x[d] * pow(x_pred[i,d] - x2[j,d], 2);
//       for (l in 1:L) c2[l] <- - beta_t[l] * pow(theta[l] - t[j,l], 2);
//       Sigma_p2[i,j] <- lambda_inv * exp(sum(c1) + sum(c2));
//     }
//   }
//   Sigma_p <- append_col(Sigma_p1, Sigma_p2);
//   Sigma_pred <- Sigma_pred0 - Sigma_p * Sigma_inv * Sigma_p';
//   Sigma_pred <- (Sigma_pred + Sigma_pred') / 2 ;
//   mu_pred <- Sigma_p * Sigma_inv * z;
//   y_rep <- multi_normal_rng(mu_pred, Sigma_pred);
// }

