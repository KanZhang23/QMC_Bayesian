clear;
absTol = 1e-2;
M = 100;
logit = @(b,x,d) exp(b(1) + sum(bsxfun(@times,b(2:d+1),x(:,1:d)),2))./...
    (1+exp(b(1) + sum(bsxfun(@times,b(2:d+1),x(:,1:d)),2)));
d = 3;
beta = -ones(1,d+1); beta(1) = 1;
beta = beta.*(1 + 0.05*randn(1,d+1));
x = -2 + 8*rand(M,d);
y = rand(M,1) < logit(beta,x,d);

getMLE;
LogLb = @(b) LogL(b,x,y,d);
Hessian = hessian(LogLb,betaMLE);
A = inv(-Hessian);
Ainv = -Hessian;
[U,S,~] = svd(A);
A0 = U*sqrt(S);

genMarPDF;