gail.InitializeWorkspaceDisplay
%%
beta = [1 -1];
M=100;
% rng(53);
logit = @(b,x) exp(bsxfun(@plus,b(1),b(2)*x))./...
          (1+exp(bsxfun(@plus,b(1),b(2)*x)));
x = linspace(-2,6,M);
y = rand(1,M) <= logit(beta,x);
getMLE;
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

%%
logitp = @(b,x) exp(b(1)+b(2)*x)./(1+exp(b(1)+b(2)*x));

prior1 = @(b) normpdf(b,0,1);    
prior2 = @(b) normpdf(b,0,1);    
post = @(b) prod(logitp(b,x).^y.*(1-logitp(b,x)).^(1-y))...  % likelihood
            .* prior1(b(1)) .* prior2(b(2));                  % priors
        
% f1 = @(b) post(b1,b2).*b(1);
% f2 = @(b) post(b1,b2).*b(2);

post_stdn = @(b) prod(logitp(b,x).^y.*(1-logitp(b,x)).^(1-y));
post_mle = @(b) post_stdn(b).*(det(Hessian))^(-0.5)...
        .*exp(-0.5*(Hessian(1,1)*(b(1)-betaMLE(1)).^2+Hessian(2,2)*(b(2)-betaMLE(2)).^2 ...
        +2*Hessian(1,2)*(b(1)-betaMLE(1)).*(b(2)-betaMLE(2))+b(1).^2+b(2).^2));
post_prod = @(b) 1.*post_stdn(b).*(det(Hessian))^(-0.5)...
        .*exp(-0.5*(Hessian(1,1)*(b(1)-betaMLE(1)).^2+Hessian(2,2)*(b(2)-betaMLE(2)).^2 ...
        +2*Hessian(1,2)*(b(1)-betaMLE(1)).*(b(2)-betaMLE(2))));
    
%% transform
post_stdn = @(b) post_stdn(gail.stdnorminv(b));
post_mle = @(b) post_mle(A0*gail.stdnorminv(b) + betaMLE);
post_prod = @(b) post_prod(A_new*gail.stdnorminv(b) + c);

nn = 100;
b1 = linspace(-3, 3, nn);
b2 = linspace(-3, 3, nn);
simpost = zeros(nn,nn);

for i = 1:length(b1)
    for j = 1:length(b2)
        simpost(i,j) = post([b1(i); b2(j)]);
        %simpost_stdn(i,j) = post_stdn([b1(i); b2(j)]);
    end;
end;
figure;
surf(b1,b2,simpost','FaceColor','red','EdgeColor','none')
camlight left; lighting phong;
xlabel('$b_0$')
ylabel('$b_1$')
xlim([-0.5 3])
ylim([-3 0.5])
ax = axis;
b1 = linspace(0, 1, nn);
b2 = linspace(0, 1, nn);


simpost_stdn = zeros(nn,nn);
simpost_mle = zeros(nn,nn);
simpost_prod = zeros(nn,nn);
for i = 1:length(b1)
    for j = 1:length(b2)
        simpost_stdn(i,j) = post_stdn([b1(i); b2(j)]);
        simpost_mle(i,j) = post_mle([b1(i); b2(j)]);
        simpost_prod(i,j) = post_prod([b1(i); b2(j)]);
    end;
end;
figure;
surf(b1,b2,simpost_stdn,'FaceColor','red','EdgeColor','none')
camlight left; lighting phong;
xlabel('$b_0$')
ylabel('$b_1$')
axis(ax)

figure;
surf(b1,b2,simpost_prod,'FaceColor','red','EdgeColor','none')
camlight left; lighting phong;
xlabel('$b_0$')
ylabel('$b_1$')

ax = axis;
figure;
surf(b1,b2,simpost_mle,'FaceColor','red','EdgeColor','none')
camlight left; lighting phong;
xlabel('$b_0$')
ylabel('$b_1$')
axis(ax)