M = 100;
logit = @(b,x,d) exp(b(1) + sum(bsxfun(@times,b(2:d+1),x(:,1:d)),2))./...
    (1+exp(b(1) + sum(bsxfun(@times,b(2:d+1),x(:,1:d)),2)));
d = 50;
b = -ones(1,d+1); b(1) = 1;
x = rand(M,d);
logit(b,x,d);