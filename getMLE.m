fun = @(b)BayesianMLE(b,x,y);
betaMLE=fsolve(fun,[0 0]);
betaMLE = betaMLE';