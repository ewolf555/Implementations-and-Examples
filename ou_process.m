function X_t = ou_process(T, n, c)


dt = (1/n);
T_grid = 0:dt:T;

X = zeros(1, length(T_grid));
W = zeros(1, length(T_grid));
X_init = 0;
W_init = 0;
X(1,1) = X_init;
W(1,1) = W_init;

for i = 2:length(T_grid)
    
    % Simulate \int_0^t exp(-cs)dW(s)
    W(1,i) = W(1,i-1) + exp(-c*T_grid(i))*normrnd(0,dt);
    
    % Full Process
    X(1,i) = exp(c*T_grid(i))*W(1,i);
    
end

X_t = X;
end