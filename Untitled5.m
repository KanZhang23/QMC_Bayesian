n_test = 2^16;
d = 2;
b = rand(n_test,d+1);
absTol = 1e-3;
M = 100;
logit = @(b,x,d) exp(b(1) + sum(bsxfun(@times,b(2:d+1),x(:,1:d)),2))./...
    (1+exp(b(1) + sum(bsxfun(@times,b(2:d+1),x(:,1:d)),2)));
beta = -ones(1,d+1); beta(1) = 1;
%x = rand(M,d);
x = (linspace(-2,6,M))'/d;
x = repmat(x,1,d);
y = rand(M,1) < logit(beta,x,d);
post = @(b) prod(bsxfun(@power,logitp(b,x,d),y').*bsxfun(@power,(1-logitp(b,x,d)),1-y'),2);
f1 = @(b) post(b).*b(:,1);
tic
f1(b);
toc
