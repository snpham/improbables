% ASEN 6519 Unicertainty Quantification
% Simulation of random variables and Gaussian random vectors
% Author: Alireza Doostan
% Spring 2014
clear all
close all

%%
% simulation of an exponential random variable
% \lambda=2
lambda = 2;

% number of samples
n_samp = 100000;

U = rand(n_samp,1);% generate independent samples from U(0,1)
X = -lambda*log(U);

% plot the empirical CDF of X_1
figure; 
cdfplot(X)

% superpose the exact CDF
cdf_exp = cdf('exp',0:.5:20,2);
hold on; plot(0:.5:20,cdf_exp,'ro','LineWidth',2)
legend('From samples','Exact')
set(gca,'FontSize',18)

%%
% simulation of a discrete random variable X=(1,2,3,4)
%with probabilities p=(1/4,1/4,1/3,1/6)
clear all
close all

% number of samples
n_samp = 1000000;

% form the CDF
F_X = [1/4 1/2 5/6 1];

U = rand(n_samp,1);% generate independent samples from U(0,1)

% Inverse CDF
for i=1:n_samp
    X(i) = find(F_X>U(i),1);
end

% count the number of occurence of (1,2,3,4) and compute the probability
p_hat = zeros(1,4);
for i=1:4
p_hat(i) = length(find(X==i))/n_samp;
end

p_hat


%%
% simulation of a Gaussian random n-vector
clear all
close all

% size of the vector
n = 10;

% number of samples
n_samp = 100000;

% define the mean \mu
mu = ones(n,1);

% generate a covariance matrix (exponential funciton)
% C(\tau)=sigma^2 exp(-|\tau|/\ell) and \tau = i-j
sigma = 1;
ell = 3;
C = zeros(n,n);

for i=1:n
    C(i,:)= sigma^2*exp(-abs([1:n]-i).^1/ell);
end

[Phi,Lambda]=eig(C);

figure; plot(Phi(:,7:10))

Y = randn(n,n_samp);
X = mu*ones(1,n_samp) + Phi*Lambda^(1/2)*Y;

abs(mean(X,2)-mu)./mu
abs(cov(X')-C)
