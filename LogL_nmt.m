function F = LogL(b1,b,x,y,d)
expxb = exp(b1 + sum(bsxfun(@times,b(1:d),x(:,1:d)),2));
oovexp = 1./(1+1./expxb);
F = sum(y.*log(oovexp) + (1-y).*log(1-oovexp));