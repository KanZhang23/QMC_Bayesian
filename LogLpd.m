function F = LogLpd(theta, X, SigmaInv)
tmp = bsxfun(@minus, X, theta) * (SigmaInv + SigmaInv');
F = sum(-0.5 * tmp);