data {
    int N; // 样本数量
    int N2; // new_X 矩阵大小
    int K; // 模型矩阵的列数
    real y[N]; // 响应变量y
    matrix[N, K] X; // 模型矩阵X
    matrix[N2, K] new_X; // 预测值矩阵
  }
parameters {
  vector[K] beta; // 回归模型中的参数
  real sigma; // 标准差
}
transformed parameters {
  vector[N] linpred;
  linpred = X*beta;
}
model {  
  beta[1] ~ cauchy(0, 10); // 截距项的先验 
  
  for(i in 2:K)
    beta[i] ~ cauchy(0, 2.5); // 斜率项的先验 
  
  y ~ normal(linpred, sigma);
}
generated quantities {
  vector[N2] y_pred;
  y_pred = new_X*beta; /// 模型预测的y的值
}
