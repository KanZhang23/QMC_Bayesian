function F = normalLogL_nmt(theta, theta1, X, SigmaInv,d)
theta = [theta1, theta];
F = sum(sum(0.5*log(det(SigmaInv)) - 0.5*d*log(2*pi)- 0.5 * bsxfun(@minus, X, theta) * SigmaInv .* bsxfun(@minus, X, theta),2));