%经验分布
[f,xi]=ksdensity(log_returns);
%对数正态分布
norm_mean = mean(log_returns);
norm_std = std(log_returns);
norm_pdf=normpdf(xi,norm_mean,norm_std);
%norm_rand=normrnd(norm_mean,norm_std,[1000000,1]);
%[f_norm,xi_norm]=ksdensity(norm_rand);
%稳定分布
para=stblfit(log_returns);
stable_pdf = stblpdf(xi,para(1),para(2),para(3),para(4));
%stbl_rand=stblrnd(para(1),para(2),para(3),para(4),[1000000,1]);
%[f_stable,xi_stable]=ksdensity(stbl_rand);
%混合分布
AIC=zeros(4,1);
BIC=zeros(4,1);
for i=2:5
GMM=fitgmdist(log_returns,i);
aic=GMM.AIC;
AIC(i-1)=aic;
bic=GMM.BIC;
BIC(i-1)=bic;
end
[m1 n1]=min(AIC);
[m2 n2]=min(BIC);
k1=n1+1;
k2=n2+1;
GMM1=fitgmdist(log_returns,k1);
GMM2=fitgmdist(log_returns,k2);
gmmpdf1=pdf(GMM1,xi')
gmmpdf2=pdf(GMM2,xi')

Y1=random(GMM1,10000000);
Y2=random(GMM2,10000000);
[f_gmm1,xi_gmm1]=ksdensity(Y1);
[f_gmm2,xi_gmm2]=ksdensity(Y2);

figure()
plot(xi,f,'blue','LineWidth',2) 
hold on 
plot(xi,stable_pdf,'red','LineWidth',2)
%plot(xi_stable,f_stable,'red','LineWidth',2)
legend('Empirical','Stable')
title('StableDistribution');
hold off

figure()
plot(xi,f,'green') 
hold on 
plot(xi,norm_pdf,'black');
%plot(xi_norm,f_norm,'black')
plot(xi,stable_pdf,'blue')
plot(xi,gmmpdf1,'Color','red');
plot(xi,gmmpdf2,'Color',[0.3 0.8 0.9]);
plot(xi_gmm1,f_gmm1,'red');
plot(xi_gmm2,f_gmm2,'Color',[0.3 0.8 0.9])
legend('Empirical','LogNorm','Stable','GaussianMixture-AIC','GaussianMixture-BIC')
title('Summary');



