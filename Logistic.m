gail.InitializeWorkspaceDisplay
beta = [1 -1];
% rng(53);
absTol = 0;
relTol = 1e-2;
M = 100;
hyperbox = [-inf(1,2);inf(1,2)];
in_param.measure = 'normal';
in_param.abstol = 0;
in_param.reltol = 1e-12;

logit = @(b,x) exp(bsxfun(@plus,b(1),b(2)*x))./...
          (1+exp(bsxfun(@plus,b(1),b(2)*x)));

logitp = @(b,x) exp(bsxfun(@plus,b(:,1),b(:,2)*x))./...
          (1+exp(bsxfun(@plus,b(:,1),b(:,2)*x)));

n = 10;
betaSobol = zeros(n,2);
for i = 1:n

x = linspace(-2,6,M);
y = rand(1,M) <= logit(beta,x);

post = @(b) prod(bsxfun(@power,logitp(b,x),y).*...
        bsxfun(@power,(1-logitp(b,x)),(1-y)),2);
f1 = @(b) post(b).*b(:,1);
f2 = @(b) post(b).*b(:,2);

in_param.mmax = 10;
for j = 1:20    
    
    [temp1, outParaSobol1] = cubSobol_g(f1,hyperbox,in_param);
    [temp2, outParaSobol2] = cubSobol_g(f2,hyperbox,in_param);
    [temp, outParaSobol] = cubSobol_g(post,hyperbox,in_param);
    v1_pm(1) = (temp1-outParaSobol1.bound_err)/(temp-outParaSobol.bound_err);
    v1_pm(2) = (temp1+outParaSobol1.bound_err)/(temp-outParaSobol.bound_err);
    v1_pm(3) = (temp1-outParaSobol1.bound_err)/(temp+outParaSobol.bound_err);
    v1_pm(4) = (temp1+outParaSobol1.bound_err)/(temp+outParaSobol.bound_err);
    v1_plus = max(v1_pm);
    v1_minus = min(v1_pm);
    v1_hat = abs(v1_plus*v1_minus)*(sign(v1_plus)+sign(v1_minus))/...
             (abs(v1_plus)+abs(v1_minus));
    tol1 = (v1_plus-v1_minus)^2/((relTol*abs(v1_plus)+relTol*abs(v1_minus))^2);
    v2_pm(1) = (temp2-outParaSobol2.bound_err)/(temp-outParaSobol.bound_err);
    v2_pm(2) = (temp2+outParaSobol2.bound_err)/(temp-outParaSobol.bound_err);
    v2_pm(3) = (temp2-outParaSobol2.bound_err)/(temp+outParaSobol.bound_err);
    v2_pm(4) = (temp2+outParaSobol2.bound_err)/(temp+outParaSobol.bound_err);
    v2_plus = max(v2_pm);
    v2_minus = min(v2_pm);
    v2_hat = abs(v2_plus*v2_minus)*(sign(v2_plus)+sign(v2_minus))/...
             (abs(v2_plus)+abs(v2_minus));
    tol2 = (v2_plus-v2_minus)^2/((relTol*abs(v2_plus)+relTol*abs(v2_minus))^2);
    if tol1 > 1 || tol2 > 1
        in_param.mmax = in_param.mmax +1;
    else
        break
    end
end
    betaSobol(i,:) = [v1_hat,v2_hat];
end
plot(betaSobol(:,1),betaSobol(:,2),'.','MarkerSize',18)
title('cubSobol');
