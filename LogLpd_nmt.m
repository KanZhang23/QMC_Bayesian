function F = LogLpd_nmt(theta, theta1, X, SigmaInv)
theta = [theta1, theta];
tmp = bsxfun(@minus, X, theta) * (SigmaInv + SigmaInv');
F = sum(-0.5 * tmp);
F(1) = [];