function F = normalLogL(theta, X, SigmaInv,d)
F = sum(sum(0.5*log(det(SigmaInv)) - 0.5*d*log(2*pi)- 0.5 * bsxfun(@minus, X, theta) * SigmaInv .* bsxfun(@minus, X, theta),2));