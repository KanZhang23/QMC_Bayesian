fun = @(theta) LogLpd_nmt(theta, theta1, X, SigmaInv);
options.MaxFunctionEvaluations = 1e5;
options.MaxIterations = 1000;
betaMLE_nmt=fsolve(fun,zeros(1,d-1),options);