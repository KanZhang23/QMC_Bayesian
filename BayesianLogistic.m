gail.InitializeWorkspaceDisplay
beta = [1 -1];
% rng(53);
absTol = 1e-3;
% relTol = 1e-2;
M = 100;

logit = @(b,x) exp(bsxfun(@plus,b(1),b(2)*x))./...
          (1+exp(bsxfun(@plus,b(1),b(2)*x)));

logitp = @(b,x) exp(bsxfun(@plus,b(:,1),bsxfun(@times,b(:,2),x)))./...
          (1+exp(bsxfun(@plus,b(:,1),bsxfun(@times,b(:,2),x))));

% for n = [20 50 100 200 400 800];

x = linspace(-2,6,M);
y = rand(1,M) <= logit(beta,x);
getMLE;

% post = @(b) bsxfun(@power,logitp(b,x),y).*...
%         bsxfun(@power,(1-logitp(b,x)),(1-y));
post = @(b) prod(bsxfun(@power,logitp(b,x),y).*bsxfun(@power,(1-logitp(b,x)),(1-y)),2);
% post = @(b) prod(post(b),2);

f1 = @(b) post(b).*b(:,1);
f2 = @(b) post(b).*b(:,2);

syms b0 b1
f = sum(y.*log(exp(b0+x*b1)./(1+exp(b0+x*b1)))+(1-y).*log(exp(b0+x*b1)./(1+exp(b0+x*b1))));
g=hessian(f,[b0,b1]);
gNum = matlabFunction(g);

Hessian = gNum(betaMLE(1),betaMLE(2));
A = inv(-Hessian);
Ainv = -Hessian;
B = eye(2);
C = inv(Ainv + B);
c = C*Ainv*betaMLE;
zc = 0.5/pi*sqrt(det(C)/(det(A)))*exp(0.5*(c'*(Ainv+B)*c-betaMLE'*Ainv*betaMLE));

[U,S,~] = svd(A);
A0 = U*sqrt(S);
[U,S,~] = svd(C);
A_new = U*sqrt(S);

f1_mle = @(b) post(b).*b(:,1).*(det(Hessian))^(-0.5)...
        .*exp(-0.5*(Hessian(1,1)*(b(:,1)-betaMLE(1)).^2+Hessian(2,2)*(b(:,2)-betaMLE(2)).^2 ...
        +2*Hessian(1,2)*(b(:,1)-betaMLE(1)).*(b(:,2)-betaMLE(2))+b(:,1).^2+b(:,2).^2));
f2_mle = @(b) post(b).*b(:,2).*(det(Hessian))^(-0.5)...
        .*exp(-0.5*(Hessian(1,1)*(b(:,1)-betaMLE(1)).^2+Hessian(2,2)*(b(:,2)-betaMLE(2)).^2 ...
        +2*Hessian(1,2)*(b(:,1)-betaMLE(1)).*(b(:,2)-betaMLE(2))+b(:,1).^2+b(:,2).^2));
post_mle = @(b) post(b).*(det(Hessian))^(-0.5)...
        .*exp(-0.5*(Hessian(1,1)*(b(:,1)-betaMLE(1)).^2+Hessian(2,2)*(b(:,2)-betaMLE(2)).^2 ...
        +2*Hessian(1,2)*(b(:,1)-betaMLE(1)).*(b(:,2)-betaMLE(2))+b(:,1).^2+b(:,2).^2));
    
f1_prod = @(b) zc.*post(b).*b(:,1).*(det(Hessian))^(-0.5)...
        .*exp(-0.5*(Hessian(1,1)*(b(:,1)-betaMLE(1)).^2+Hessian(2,2)*(b(:,2)-betaMLE(2)).^2 ...
        +2*Hessian(1,2)*(b(:,1)-betaMLE(1)).*(b(:,2)-betaMLE(2))));
f2_prod = @(b) zc.*post(b).*b(:,2).*(det(Hessian))^(-0.5)...
        .*exp(-0.5*(Hessian(1,1)*(b(:,1)-betaMLE(1)).^2+Hessian(2,2)*(b(:,2)-betaMLE(2)).^2 ...
        +2*Hessian(1,2)*(b(:,1)-betaMLE(1)).*(b(:,2)-betaMLE(2))));
post_prod = @(b) zc.*post(b).*(det(Hessian))^(-0.5)...
        .*exp(-0.5*(Hessian(1,1)*(b(:,1)-betaMLE(1)).^2+Hessian(2,2)*(b(:,2)-betaMLE(2)).^2 ...
        +2*Hessian(1,2)*(b(:,1)-betaMLE(1)).*(b(:,2)-betaMLE(2))));

n = 50;
betaSobol = zeros(n,2);
betaSobol_mle = zeros(n,2);    
betaSobol_prod = zeros(n,2);
betaMCMC = zeros(n,2);
qmn = zeros(15,n);
qmn_mle = zeros(15,n);
qmn_prod = zeros(15,n);
for i = 1:n
    [q1,out_param1,qm1] = cubSobolBayesian(f1,post,absTol);
    [q2,out_param2,qm2] = cubSobolBayesian(f2,post,absTol);
    qmn(1:length(qm1),i) = qm1;
    Nqmn(i) = nnz(qmn(:,i));
    Nmax = min(Nqmn);
    betaSobol(i,:) = [q1,q2];
    [q1_mle,out_param1_mle,qm_mle1] = cubSobolBayesian_IS(f1_mle,post_mle,absTol,A0,betaMLE);
    [q2_mle,out_param2_mle,~] = cubSobolBayesian_IS(f2_mle,post_mle,absTol,A0,betaMLE);
    qmn_mle(1:length(qm_mle1),i) = qm_mle1;
    Nqmn_mle(i) = nnz(qmn_mle(:,i));
    Nmax_mle = min(Nqmn_mle);
    betaSobol_mle(i,:) = [q1_mle,q2_mle];
    
    [q1_prod,out_param1_prod,qm_prod1] = cubSobolBayesian_IS(f1_prod,post_prod,absTol,A_new,c);
    [q2_prod,out_param2_prod,~] = cubSobolBayesian_IS(f2_prod,post_prod,absTol,A_new,c);
    qmn_prod(1:length(qm_prod1),i) = qm_prod1;
    Nqmn_prod(i) = nnz(qmn_prod(:,i));
    Nmax_prod = min(Nqmn_prod);
    betaSobol_prod(i,:) = [q1_prod,q2_prod];    
    
%     slicesampler;
%     betaMCMC(i,:) = mean(MCMCsample);
end
corner_sw = min([betaSobol;betaSobol_mle;betaSobol_prod]);
corner_ne = max([betaSobol;betaSobol_mle;betaSobol_prod]);
center = 0.5*(corner_sw + corner_ne);
corner = center-absTol;

%% MCMC


%%
figure;
% plot(betaSobol(:,1),betaSobol(:,2),'o',betaSobol_mle(:,1),betaSobol_mle(:,2),...
%     '+',betaSobol_prod(:,1),betaSobol_prod(:,2),'*',betaMCMC(:,1),betaMCMC(:,2),'^','MarkerSize',10)
plot(betaSobol(:,1),betaSobol(:,2),'o',betaSobol_mle(:,1),betaSobol_mle(:,2),...
    '+',betaSobol_prod(:,1),betaSobol_prod(:,2),'*','MarkerSize',10)
hold on;
rectangle('position',[corner 2*absTol 2*absTol],'EdgeColor','r','LineWidth',1.5)
title([num2str(n),' replications, tol = ',num2str(absTol)]);
xlabel('$\hat{\beta}_1$');
ylabel('$\hat{\beta}_2$');
axis equal;
legend('$\pi$','$\rho_{MLE}$','$\pi\cdot\rho_{MLE}$','MCMC','Location','southwest')
legend('boxon')


% n=5;
% figure;
% if Nmax >5 
%     Nmax =5;
% end
% for i = 1:Nmax
% plot(zeros(1,n),qmn(i,1:n),'o','Markersize',10)
% hold on;
% end
% title('prior');
% legend('n=1024','n=2048','n=4096','n=8192','n=16384');
% % figure;
% if Nmax_mle >5 
%     Nmax_mle =5;
% end
% for i = 1:Nmax_mle
% plot(-.5*zeros(1,n),qmn_mle(i,1:n),'o','Markersize',10)
% hold on;
% end
% title('MLE');
% legend('n=1024','n=2048','n=4096','n=8192','n=16384');
% % figure;
% if Nmax_prod >5 
%     Nmax_prod =5;
% end
% for i = 1:Nmax_prod
% plot(0.5*zeros(1,n),qmn_prod(i,1:n),'o','Markersize',10)
% hold on;
% end
% title('prior*MLE');
% legend('n=1024','n=2048','n=4096','n=8192','n=16384');
% % end