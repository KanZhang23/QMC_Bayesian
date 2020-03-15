fun_nmt = @(b)BayesianMLE_nmt(b1,b,x,y,d);
% options = optimoptions('fsolve','Display','iter');
options.MaxFunctionEvaluations = 1e5;
options.MaxIterations = 1000;
betaMLE_nmt=fsolve(fun_nmt,zeros(1,d),options);
% betaMLE = betaMLE';