clear;
absTol = 1e-2;
d = 2;
n = 10;
mu = zeros(1,d); Omega = generateSPDmatrix(d); OmegaInv = inv(Omega);
theta = ones(1,d); 
for i = 1:d
    if mod(i,2) == 0
        theta(i) = -1;
    end
end
Sigma = generateSPDmatrix(d); SigmaInv = inv(Sigma);
X = mvnrnd(theta, Sigma, n);
normal_getMLE;
normal_LogLtheta = @(theta) normalLogL(theta, X, SigmaInv,d);
Hessian = hessian(normal_LogLtheta,betaMLE);
A = inv(-Hessian);
AInv = -Hessian;
[U,S,~] = svd(A);
A0 = U*sqrt(S);

v_theta1 = -2:0.2:4;
i_theta1 = 0;
for theta1 = v_theta1
    i_theta1 = i_theta1 +1;
normal_getMLE_nmt;
normal_LogLtheta_nmt = @(theta) normalLogL_nmt(theta, theta1, X, SigmaInv,d);
Hessian_nmt = hessian(normal_LogLtheta_nmt,betaMLE_nmt);
A_nmt = inv(-Hessian_nmt);
AInv_nmt = -Hessian_nmt;
[U_nmt,S_nmt,~] = svd(A_nmt);
A0_nmt = U_nmt*sqrt(S_nmt);

t_start = tic;
%% Initial important cone factors and Check-initialize parameters
r_lag = 4; %distance between coefficients summed and those computed
% l_star = out_param.mmin - r_lag; % Minimum gathering of points for the sums of DFWT
% omg_circ = @(m) 2.^(-m);
% omg_hat = @(m) out_param.fudge(m)/((1+out_param.fudge(r_lag))*omg_circ(r_lag));

out_param.d = d;
out_param.mmin = 10;
out_param.mmax = 22;
out_param.fudge = @(m) 5*2.^-m;
%% Main algorithm
sobstr=sobolset(out_param.d); %generate a Sobol' sequence
sobstr=scramble(sobstr,'MatousekAffineOwen'); %scramble it
sobstr_nmt=sobolset(d-1); %generate a Sobol' sequence
sobstr_nmt=scramble(sobstr_nmt,'MatousekAffineOwen'); %scramble it

Stilde1=zeros(out_param.mmax-out_param.mmin+1,1); %initialize sum of DFWT terms
Stilde2=zeros(out_param.mmax-out_param.mmin+1,1);
% CStilde_low = -inf(1,out_param.mmax-l_star+1); %initialize various sums of DFWT terms for necessary conditions
% CStilde_up = inf(1,out_param.mmax-l_star+1); %initialize various sums of DFWT terms for necessary conditions
errest1=zeros(out_param.mmax-out_param.mmin+1,1); %initialize error estimates
errest2=zeros(out_param.mmax-out_param.mmin+1,1);
appxinteg1=zeros(out_param.mmax-out_param.mmin+1,1); %initialize approximations to integral
appxinteg2=zeros(out_param.mmax-out_param.mmin+1,1);
exit_len = 2;
out_param.exit=false(1,exit_len); %we start the algorithm with all warning flags down
%% Initial points and FWT
out_param.n=2^out_param.mmin; %total number of points to start with
n0=out_param.n; %initial number of points
xpts=sobstr(1:n0,1:out_param.d); %grab Sobol' points
xpts_nmt=sobstr_nmt(1:n0,1:d-1); %grab Sobol' points
b_nmt = bsxfun(@plus,gail.stdnorminv(xpts_nmt)*A0_nmt', betaMLE_nmt);
b_nmt_aug = [theta1*ones(n0,1), b_nmt];
b = bsxfun(@plus,gail.stdnorminv(xpts)*A0', betaMLE);
for i_n0 = 1:n0
    L(i_n0,1) = prod((2*pi)^(-d/2)*sqrt(det(SigmaInv))*sum(exp(-0.5*bsxfun(@minus, X, b(i_n0,:)) * SigmaInv .* bsxfun(@minus, X, b(i_n0,:))), 2));
end
prior = (2*pi)^(-d/2)*sqrt(det(OmegaInv))*sum(exp(-0.5*bsxfun(@minus, mu, b) * OmegaInv .* bsxfun(@minus, mu, b)), 2);
normal_IS = (2*pi)^(-d/2)*sqrt(det(AInv))*sum(exp(-0.5*bsxfun(@minus, betaMLE, b) * AInv .* bsxfun(@minus, betaMLE, b)), 2);
y2 = L .* prior ./ normal_IS;
for i_n0 = 1:n0
    L_nmt(i_n0,1) = prod((2*pi)^(-d/2)*sqrt(det(SigmaInv))*sum(exp(-0.5*bsxfun(@minus, X, b_nmt_aug(i_n0,:)) * SigmaInv .* bsxfun(@minus, X, b_nmt_aug(i_n0,:))), 2));
end
prior_nmt = (2*pi)^(-d/2)*sqrt(det(OmegaInv))*sum(exp(-0.5*bsxfun(@minus, mu, b_nmt_aug) * OmegaInv .* bsxfun(@minus, mu, b_nmt_aug)), 2);
normal_IS_nmt = (2*pi)^(-(d-1)/2)*sqrt(det(AInv_nmt))*sum(exp(-0.5*bsxfun(@minus, betaMLE_nmt, b_nmt) * AInv_nmt .* bsxfun(@minus, betaMLE_nmt, b_nmt)), 2);
y1 = L_nmt .* prior_nmt ./ normal_IS_nmt;
yval1 = y1;
yval2 = y2;
%% Compute initial FWT
for l=0:out_param.mmin-1
   nl=2^l;
   nmminlm1=2^(out_param.mmin-l-1);
   ptind=repmat([true(nl,1); false(nl,1)],nmminlm1,1);
   evenval1=y1(ptind);
   oddval1=y1(~ptind);
   evenval2=y2(ptind);
   oddval2=y2(~ptind);
   y1(ptind)=(evenval1+oddval1)/2;
   y1(~ptind)=(evenval1-oddval1)/2;
   y2(ptind)=(evenval2+oddval2)/2;
   y2(~ptind)=(evenval2-oddval2)/2;   
end
%y now contains the FWT coefficients

%% Create kappanumap implicitly from the data
kappanumap1=(1:out_param.n)'; %initialize map
kappanumap2=(1:out_param.n)';
for l=out_param.mmin-1:-1:1
   nl=2^l;
   oldone1=abs(y1(kappanumap1(2:nl))); %earlier values of kappa, don't touch first one
   newone1=abs(y1(kappanumap1(nl+2:2*nl))); %later values of kappa, 
   oldone2=abs(y1(kappanumap2(2:nl))); %earlier values of kappa, don't touch first one
   newone2=abs(y1(kappanumap2(nl+2:2*nl))); %later values of kappa, 
   flip1=find(newone1>oldone1); %which in the pair are the larger ones
   flip2=find(newone2>oldone2); %which in the pair are the larger ones
   if ~isempty(flip1)
       flipall=bsxfun(@plus,flip1,0:2^(l+1):2^out_param.mmin-1);
       flipall=flipall(:);
       temp=kappanumap1(nl+1+flipall); %then flip 
       kappanumap1(nl+1+flipall)=kappanumap1(1+flipall); %them
       kappanumap1(1+flipall)=temp; %around
   end
   if ~isempty(flip2)
       flipall=bsxfun(@plus,flip2,0:2^(l+1):2^out_param.mmin-1);
       flipall=flipall(:);
       temp=kappanumap2(nl+1+flipall); %then flip 
       kappanumap2(nl+1+flipall)=kappanumap2(1+flipall); %them
       kappanumap2(1+flipall)=temp; %around
   end
end

%% Compute Stilde
nllstart = int64(2^(out_param.mmin-r_lag-1));
Stilde1(1)=sum(abs(y1(kappanumap1(nllstart+1:2*nllstart))));
out_param.bound_err1=out_param.fudge(out_param.mmin)*Stilde1(1);
errest1(1)=out_param.bound_err1;

Stilde2(1)=sum(abs(y2(kappanumap2(nllstart+1:2*nllstart))));
out_param.bound_err2=out_param.fudge(out_param.mmin)*Stilde2(1);
errest2(1)=out_param.bound_err2;


%% Approximate integral
q1=mean(yval1);
appxinteg1(1)=q1;
q2=mean(yval2);
appxinteg2(1)=q2;

% Check the end of the algorithm
if (appxinteg2(1)-errest2(1))*(appxinteg2(1)+errest2(1))<0
    warning('The range of denominator contains 0')
end

v_pm(1) = (appxinteg1(1)-errest1(1))/((appxinteg2(1)-errest2(1)));
v_pm(2) = (appxinteg1(1)+errest1(1))/((appxinteg2(1)-errest2(1)));
v_pm(3) = (appxinteg1(1)-errest1(1))/((appxinteg2(1)+errest2(1)));
v_pm(4) = (appxinteg1(1)+errest1(1))/((appxinteg2(1)+errest2(1)));

v_plus = max(v_pm);
v_minus = min(v_pm);
v_hat =  (v_plus + v_minus)/2;
qm(1,1) = v_hat;
tol = (v_plus-v_minus)^2/((2*absTol)^2);
is_done = false;
if tol <= 1
   q=v_hat;
   q_sim=q1/q2;
   out_param.time=toc(t_start);
   is_done = true;
end
out_param.errbound = (v_plus - v_minus)/2;
p_theta1(i_theta1) = v_hat;
err_p_theta1(i_theta1) = out_param.errbound;
disp([num2str(v_hat), ' +- ',num2str(out_param.errbound)])
end
figure;
plot(v_theta1,p_theta1)
xlabel('\theta_1')
title('marginal PDF of the fisrt parameter')