clear;
absTol = 1e-2;
M = 100;
logit = @(b,x,d) exp(b(1) + sum(bsxfun(@times,b(2:d+1),x(:,1:d)),2))./...
    (1+exp(b(1) + sum(bsxfun(@times,b(2:d+1),x(:,1:d)),2)));
d = 4;
beta = -ones(1,d+1); beta(1) = 1;
beta = beta.*(1 + 0.05*randn(1,d+1));
x = -2 + 8*rand(M,d);
y = rand(M,1) < logit(beta,x,d);
getMLE;
post = @(b) prod(bsxfun(@power,logitp(b,x,d),y').*bsxfun(@power,(1-logitp(b,x,d)),1-y'),2);
f1 = @(b) post(b).*b(:,1);
f2 = @(b) post(b).*b(:,2);
LogLb = @(b) LogL(b,x,y,d);
Hessian = hessian(LogLb,betaMLE);
A = inv(-Hessian);
Ainv = -Hessian;
[U,S,~] = svd(A);
A0 = U*sqrt(S);

post_mle = @(b) post(b).*(det(-Hessian))^(-0.5)...
            .*exp(-0.5*(sum((bsxfun(@minus,b,betaMLE)*Hessian).*bsxfun(@minus,b,betaMLE),2)+...
                  sum(b.*b,2)));
f1_mle = @(b) post_mle(b).*b(:,1);
f2_mle = @(b) post_mle(b).*b(:,2);

n = 8;
betaSobol = zeros(n,d);
betaSobol_mle = zeros(n,d);    
betaSobol_prod = zeros(n,d);
betaSobol_s = zeros(n,d);
betaSobol_mle_s = zeros(n,d);    
betaSobol_prod_s = zeros(n,d);

for i = 1:n
    disp(['The ', num2str(i), '-th replication']);
    [q1,q1_s,out_param1,qm1] = cubSobolBayesian(f1,post,absTol,d);
    [q2,q2_s,out_param2,qm2] = cubSobolBayesian(f2,post,absTol,d);
    qmn(1:length(qm1),i) = qm1;
    Nqmn(i) = nnz(qmn(:,i));
    Nmax = min(Nqmn);
    betaSobol(i,1:2) = [q1,q2];
    betaSobol_s(i,1:2) = [q1_s,q2_s];
    
    [q1_mle,q1_mle_s,out_param1_mle,qm_mle1] = cubSobolBayesian_IS(f1_mle,post_mle,absTol,A0,betaMLE,d);
    [q2_mle,q2_mle_s,out_param2_mle,~] = cubSobolBayesian_IS(f2_mle,post_mle,absTol,A0,betaMLE,d);
    qmn_mle(1:length(qm_mle1),i) = qm_mle1;
    Nqmn_mle(i) = nnz(qmn_mle(:,i));
    Nmax_mle = min(Nqmn_mle);
    betaSobol_mle(i,1:2) = [q1_mle,q2_mle];
    betaSobol_mle_s(i,1:2) = [q1_mle_s,q2_mle_s];
end
corner_sw = min([betaSobol(:,1:2);betaSobol_mle(:,1:2)]);
corner_ne = max([betaSobol(:,1:2);betaSobol_mle(:,1:2)]);
% corner_sw = min([betaSobol(:,1:2)]);
% corner_ne = max([betaSobol(:,1:2)]);
% corner_sw = min([betaSobol_mle(:,1:2)]);
% corner_ne = max([betaSobol_mle(:,1:2)]);
center = 0.5*(corner_sw + corner_ne);
corner = center-absTol;

%% 
disp(['Calculating the first component of beta_hat via standard normal density uses ',num2str(out_param1.n), ' samples, takes ', num2str(out_param1.time), 's.'])
disp(['Calculating the first component of beta_hat vis MLE density uses ',num2str(out_param1_mle.n), ' samples, takes ', num2str(out_param1_mle.time), 's.'])
% disp(['Calculating the first component of beta_hat via the product uses ',num2str(out_param1_prod.n), ' samples, takes ', num2str(out_param1_prod.time), 's.'])

%%
figure;
plot(betaSobol(:,1),betaSobol(:,2),'o',betaSobol_mle(:,1),betaSobol_mle(:,2),...
    '+','MarkerSize',10);
% plot(betaSobol(:,1),betaSobol(:,2),'o','MarkerSize',10);
% plot(betaSobol_mle(:,1),betaSobol_mle(:,2),'o','MarkerSize',10);
hold on;
rectangle('position',[corner 2*absTol 2*absTol],'EdgeColor','r','LineWidth',1.5);
