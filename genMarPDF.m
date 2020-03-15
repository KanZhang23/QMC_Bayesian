i_b1 = 0;
v_b1 = -4:0.4:6;
for b1 = v_b1
i_b1 = i_b1 + 1;
getMLE_nmt;
% post = @(b) prod(bsxfun(@power,logitp(b,x,d),y').*bsxfun(@power,(1-logitp(b,x,d)),1-y'),2);
LogLb_nmt = @(b) LogL_nmt(b1,b,x,y,d);
Hessian_nmt = hessian(LogLb_nmt,betaMLE_nmt);
A_nmt = inv(-Hessian_nmt);
Ainv_nmt = -Hessian_nmt;
[U_nmt,S_nmt,~] = svd(A_nmt);
A0_nmt = U_nmt*sqrt(S_nmt);

t_start = tic;
%% Initial important cone factors and Check-initialize parameters
r_lag = 4; %distance between coefficients summed and those computed
% l_star = out_param.mmin - r_lag; % Minimum gathering of points for the sums of DFWT
% omg_circ = @(m) 2.^(-m);
% omg_hat = @(m) out_param.fudge(m)/((1+out_param.fudge(r_lag))*omg_circ(r_lag));

out_param.d = d+1;
out_param.mmin = 10;
out_param.mmax = 22;
out_param.fudge = @(m) 5*2.^-m;
%% Main algorithm
sobstr=sobolset(out_param.d); %generate a Sobol' sequence
sobstr=scramble(sobstr,'MatousekAffineOwen'); %scramble it
sobstr_nmt=sobolset(d); %generate a Sobol' sequence
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
xpts_nmt=sobstr_nmt(1:n0,1:d); %grab Sobol' points
b_nmt = bsxfun(@plus,gail.stdnorminv(xpts_nmt)*A0_nmt', betaMLE_nmt);
b_nmt_aug = [b1*ones(n0,1), b_nmt];
b = bsxfun(@plus,gail.stdnorminv(xpts)*A0', betaMLE);
expxb = exp(bsxfun(@plus,b(:,2:d+1)*x',b(:,1)));
lgp = 1./(1+1./expxb);
expxb_nmt = exp(b_nmt*x' + b1);
lgp_nmt = 1./(1+1./expxb_nmt);
L = prod(bsxfun(@power,lgp,y').*bsxfun(@power,(1-lgp),1-y'),2);
y2 = L.*(det(-Hessian))^(-0.5).*exp(-0.5*(sum(bsxfun(@times,bsxfun(@minus,b,betaMLE)*Ainv,(bsxfun(@minus,b,betaMLE))),2)-sum(bsxfun(@times,b,b),2)));
L_nmt = prod(bsxfun(@power,lgp_nmt,y').*bsxfun(@power,(1-lgp_nmt),1-y'),2);
y1 = L_nmt.*((2*pi)*det(-Hessian_nmt))^(-0.5).*exp(-0.5*(sum(bsxfun(@times,bsxfun(@minus,b_nmt,betaMLE_nmt)*Ainv_nmt,(bsxfun(@minus,b_nmt,betaMLE_nmt))),2)-sum(bsxfun(@times,b_nmt_aug,b_nmt_aug),2)));
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
p_b1(i_b1) = v_hat;
err_p_b1(i_b1) = out_param.errbound;
disp([num2str(v_hat), ' +- ',num2str(out_param.errbound)])
end
figure;
plot(v_b1,p_b1)
xlabel('\beta_1')
title('marginal PDF of the fisrt parameter')