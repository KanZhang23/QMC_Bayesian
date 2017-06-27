%%
beta = [1 -1];
M=100;
% rng(53);

x = linspace(-2,6,M);
logit = @(b) exp(b(1)+b(2)*x)./(1+exp(b(1)+b(2)*x));
y = rand(1,M) <= logit(beta);
%getMLE;

prior1 = @(b) normpdf(b,0,1);    
prior2 = @(b) normpdf(b,0,1);    
post = @(b) prod(logit(b).^y.*(1-logit(b)).^(1-y))...  % likelihood
            .* prior1(b(1)) .* prior2(b(2));                  % priors

start = beta;        
N = 40000;
%tic;
MCMCsample = slicesample(start,N,'pdf',post,'burnin',1000);
%toc;