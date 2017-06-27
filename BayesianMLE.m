function F = BayesianMLE(b,x,y)
expxb = exp(b(1)+x*b(2));
oovexp = 1./(1+1./expxb);
F = [sum(y./(1+expxb)-(1-y).*oovexp), ...
    sum(y.*x./(1+expxb)-x.*(1-y).*oovexp)];