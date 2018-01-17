n = 50;
betaMCMC = zeros(n,2);
for i = 1:n
    slicesampler;
    betaMCMC(i,:) = mean(MCMCsample);
end
plot(betaMCMC(:,1),betaMCMC(:,2),'^','MarkerSize',5)
title([num2str(n),' replications using MCMC ']);
xlabel('$\hat{\beta}_1$');
ylabel('$\hat{\beta}_2$');