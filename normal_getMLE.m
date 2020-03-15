fun = @(theta) LogLpd(theta, X, SigmaInv);
options.MaxFunctionEvaluations = 1e5;
options.MaxIterations = 1000;
betaMLE=fsolve(fun,zeros(1,d),options);