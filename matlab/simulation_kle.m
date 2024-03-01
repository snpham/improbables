% We want to simulate a Gaussian random process defined on [-0.5:0.5] with
% an exponential covariance function 1*exp(-abs(x_1-x_2)/l) where l is
% the correlation function
clear all
close all

% define the spatial locaitons
a=1;
t=-1*a:.005:a;

% correlation length
l=2;

% number of temrs in the KLE truncation
d=2;

for i = 0 : ceil(d/2)

    % compute interval where zeros are to be found
    intv = [max((2*i-1)*pi/(2*a)+0.00000001, 0) (2*i+1)*pi/(2*a)-0.00000001];
    % compute data associated with equation : c * tan (a*x) + x = 0 (even
    % terms)

    if ((i > 0) && (2*i <=d))
        w = fzero(@(x) (1/l)*tan(a*x)+x, intv);

        KLEIG(2 * i , 1) = w; % omega
        KLEIG(2 * i , 2) = 2*l/( w^2*l^2 + 1); % lambda
        KLEIG(2 * i , 3) = 1/sqrt(a - sin(2*w*a)/(2*w)); % coefficient of phi
    end;

    % compute data associated with equation : c - x * tan (a*x) = 0 (odd
    % terms)
    if ((2*i +1) <= d)

        w = fzero(@(x) (1/l)-x*tan(a*x) , intv);

        KLEIG(2 * i +1 , 1) = w ; % omega_n
        KLEIG(2 * i +1 , 2) = 2*l/( w^2*l^2 + 1); % lambda_n
        KLEIG(2 * i +1 , 3) = 1/sqrt(a + sin(2*w*a)/(2*w)); % coefficient of phi
    end
end

% plot samples of the process
X = zeros(1,size(t,2));

for i = 0 : ceil(d/2)
    % contributions from even terms
    if ((i > 0) && (2*i <=d))

        X = X + sqrt(KLEIG(2*i,2))*KLEIG(2*i,3)*sin(KLEIG(2*i,1)*t)*randn;
        figure (1); 
        hold on;
        plot(t,KLEIG(2*i,3)*sin(KLEIG(2*i,1)*t),'Linewidth',2); % plot the eigenfunctions
    end
% contributions from odd terms
    if ((2*i +1) <= d)

        X = X + sqrt(KLEIG(2*i+1,2))*KLEIG(2*i+1,3)*cos(KLEIG(2*i+1,1)*t)*randn;
        figure (1); 
        hold on;
        plot(t,KLEIG(2*i+1,3)*cos(KLEIG(2*i+1,1)*t),'Linewidth',2); % plot the eigenfunctions
    end
end

figure (1); 
title('Eigenfunctions')
legend()
set(gca,'FontSize',18)
figure (2);hold on;
plot(t,X,'Linewidth',2)
title('A sample of the process')
set(gca,'FontSize',18)

% check the convergence of the variance
Var = zeros(1,size(t,2));

for i = 0 : ceil(d/2)

% contributions from even terms
    if ((i > 0) && (2*i <=d))
        Var = Var + KLEIG(2*i,2)*(KLEIG(2*i,3)*sin(KLEIG(2*i,1)*t)).^2;
    end

% contributions from even terms
    if ((2*i +1) <= d)
        Var = Var + KLEIG(2*i+1,2)*(KLEIG(2*i+1,3)*cos(KLEIG(2*i+1,1)*t)).^2;
    end
end

figure (3);hold on;plot(t,Var,'Linewidth',2);
title('Variance of the process')
set(gca,'FontSize',18)
figure (4);
hold on;

plot(KLEIG(:,2)/KLEIG(1,2),'ro--','Linewidth',2); % plot eigenvalues
title('Eigenvalues')
set(gca,'FontSize',18)
