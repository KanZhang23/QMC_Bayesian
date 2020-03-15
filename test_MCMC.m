npar = out_param.d;
nsimu = 2000;
data.x = x;
data.y = y;
model.ssfun = @(b,data) -2*LogL(b,data.x,data.y,d);
for i=1:npar, params{i} = {sprintf('b_{%d}',i), 0}; end
model.N = M;
options.nsimu  = nsimu;
[res,chain] = mcmcrun(model,data,params,options);
% [res(j),chain(:,:,j)] = mcmcrun(model,YObs,params,options);
% save('chain100000k.mat', 'chain', 'res');