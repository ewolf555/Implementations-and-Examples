%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                         %
%                Ornstein Uhlenbeck Process - Simulations                 %
%                                                                         % 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Definition of ann Ornstein Uhlenbeck Process as given in Hassler (2016):
%
%     X(t) = mu + + exp(ct) \int_0^t exp(-cs)dW(s) with t>=0 and X(0)=0
%         
%     W(t) is a Wiener Process with 
%
%                            E(W(t)) = 0, 
%                            Var(W(t)) = t 
%                            Cov(W(t),W(s)) = t-s
%          
%-------------------------------------------------------------------------%

% Set timesteps T, grid-size n and adjustment parameter c
T = 20;
n = 1000;
c_vals = [-0.1, -0.9];

X_t_list = zeros(length(0:1/n:T),2);

for i = 1:2
    
    X_t_list(:,i) = ou_process(T, n, c_vals(i));

end

figure;
plot(T_grid, X_t_list)
legend('c= -0.1', 'c= -0.9')   
title('Stationary Ohrnstein-Uhlenbeck Processes')
xlabel('t')
ylabel('X(t)')

