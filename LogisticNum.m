gail.InitializeWorkspaceDisplay
%%
d = 1;
beta = -ones(1,d+1); beta(1) = 1;
M=100;
% rng(53);
% logit = @(b,x) exp(bsxfun(@plus,b(1),b(2)*x))./...
%           (1+exp(bsxfun(@plus,b(1),b(2)*x)));
logit = @(b,x,d) exp(b(1) + sum(bsxfun(@times,b(2:d+1),x(:,1:d)),2))./...
    (1+exp(b(1) + sum(bsxfun(@times,b(2:d+1),x(:,1:d)),2)));
x = -2 + 8*rand(M,d);
y = rand(M,1) < logit(beta,x,d);
getMLE;
LogLb = @(b) LogL(b,x,y,d);
Hessian = hessian(LogLb,betaMLE);
A = inv(-Hessian);
Ainv = -Hessian;
% B = eye(2);
% C = inv(Ainv + B);
% c = C*Ainv*betaMLE;
% zc = 0.5/pi*sqrt(det(C)/(det(A)))*exp(0.5*(c'*(Ainv+B)*c-betaMLE'*Ainv*betaMLE));
[U,S,~] = svd(A);
A0 = U*sqrt(S);
% [U,S,~] = svd(C);
% A_new = U*sqrt(S);

%%

% prior1 = @(b) normpdf(b,0,1);    
% prior2 = @(b) normpdf(b,0,1);    
% post = @(b) prod(logitp(b,x).^y.*(1-logitp(b,x)).^(1-y))...  % likelihood
%             .* prior1(b(1)) .* prior2(b(2));                  % prior


post_stdn = @(b) prod(logitp(b,x,d).^(y').*(1-logitp(b,x,d)).^((1-y)'));
post_mle = @(b) post_stdn(b).*(det(-Hessian))^(-0.5)...
            .*exp(-0.5*((b-betaMLE)*Hessian*(b-betaMLE)'+ b*b'));

%% transform
post_stdn = @(b) post_stdn(gail.stdnorminv(b));
post_mle = @(b) post_mle(gail.stdnorminv(b)*A0' + betaMLE);
% post_prod = @(b) post_prod(A_new*gail.stdnorminv(b) + c);

nn = 100;
b1 = linspace(0, 1, nn);
b2 = linspace(0, 1, nn);
b3 = linspace(-3, 3, nn);
% simpost = zeros(nn,nn);
% 
% for i = 1:length(b1)
%     for j = 1:length(b2)
%         simpost(i,j) = post([b1(i) b2(j)]);
%         %simpost_stdn(i,j) = post_stdn([b1(i); b2(j)]);
%     end;
% end;
% figure;
% subplot(2,2,1);
% surf(b1,b2,simpost','FaceColor','red','EdgeColor','none')
% camlight left; lighting phong;
% xlabel('$\beta_1$')
% ylabel('$\beta_2$')
% xlim([-1 3])
% ylim([-3 1])
% title('$L\pi$')
% b1 = linspace(0, 1, nn);
% b2 = linspace(0, 1, nn);


simpost_stdn = zeros(nn,nn);
simpost_mle = zeros(nn,nn);
simpost_prod = zeros(nn,nn);
for i = 1:length(b1)
    for j = 1:length(b2)
%         for k = 1:length(b3)
            simpost_stdn(i,j) = post_stdn([b1(i) b2(j)]);
            simpost_mle(i,j) = post_mle([b1(i) b2(j)]);
    %         simpost_prod(i,j) = post_prod([b1(i); b2(j)]);
%         end
    end;
end;
subplot(2,2,2);
surf(b1,b2,simpost_stdn(:,:),'FaceColor','red','EdgeColor','none')
camlight left; lighting phong;
xlabel('$\beta_1$')
ylabel('$\beta_2$')
title('$f_{\pi}$')
%axis(ax)


subplot(2,2,3);
surf(b1,b2,simpost_mle(:,:),'FaceColor','red','EdgeColor','none')
camlight right; lighting phong;
xlabel('$\beta_1$')
ylabel('$\beta_2$')
title('$f_{\rho_{MLE}}$')
ax = axis;

% subplot(2,2,4);
% surf(b1,b2,simpost_prod,'FaceColor','red','EdgeColor','none')
% camlight left; lighting phong;
% xlabel('$\beta_1$')
% ylabel('$\beta_2$')
% title('$f_{\pi\rho_{MLE}}$')
% axis(ax)