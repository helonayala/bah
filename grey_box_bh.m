clc
clear
close all

addpath(genpath('casadi/'))

import casadi.*

%%%%%%%%%%%% load data %%%%%%%%%%%%%%%%%%%%%
% File to load
% Dados de treino
load('seqDegrausFixo2.mat')
u_data = u;
y_data = y;

%Dados de teste
load("malha_fechada2.mat")
u_test = u;
y_test = y;

N  = length(u_data);  % Number of samples
Ts = 0.1;  % sampling time (seconds)
fs = 1/Ts;       % Sampling frequency [hz]
t = ((1:N)-1)*Ts;

x0 = DM([y_data(1),0]); % initial condition for simulation
x0_test = DM([y_test(1), 0]);

%%%%%%%%%%%% MODELING %%%%%%%%%%%%%%%%%%%%%
om  = MX.sym('om');  % Velocidade angular
om_p = MX.sym('om_p'); % Aceleração angular
ue  = MX.sym('ue');  % Entrada

states = [om;om_p]; %vetor de estados, vel e acel
controls = ue;

In = MX.sym('In'); %Inércia
kn = MX.sym('kn'); % Ganho
bn   = MX.sym('bn'); %viscosidade
Fcn = MX.sym('Fcn'); %Atrito de Coulomb

params   = [In;kn;bn; Fcn];
parammax = [290; 550; 50; 40]; 
parammin = [0; 10; 0; 0];

nparam = length(params);
rng(42);
param_guess = rand(nparam,1);

lbx = zeros(nparam,1);
ubx = ones(nparam,1);

I    = denorm(In,   parammax(1),parammin(1));
k   = denorm(kn,  parammax(2),parammin(2));
b   = denorm(bn,  parammax(3),parammin(3));
Fc   = denorm(Fcn,  parammax(4),parammin(4));

Ff = Fc*sign(om);

rhs = [om_p; (k*ue - b*om - Ff)/I];

% Form an ode function
ode = Function('ode',{states,controls,params},{rhs});

%% helper functions
function v = denorm(vn,vmax,vmin)
    v = vmin + (vmax-vmin)*vn;
end
function vn = normalize(v,vmax,vmin)
    vn = (v - vmin) ./ (vmax-vmin);
end

%%%%%%%%%%%% Creating a simulator %%%%%%%%%%
N_steps_per_sample = 10;
dt = 1/fs/N_steps_per_sample;

% Build an integrator for this system: Runge Kutta 4 integrator
k1 = ode(states,controls,params);
k2 = ode(states+dt/2.0*k1,controls,params);
k3 = ode(states+dt/2.0*k2,controls,params);
k4 = ode(states+dt*k3,controls,params);

states_final = states+dt/6.0*(k1+2*k2+2*k3+k4);

% Create a function that simulates one step propagation in a sample
one_step = Function('one_step',{states, controls, params},{states_final});

X = states;
for i=1:N_steps_per_sample
    X = one_step(X, controls, params);
end
%
% % Create a function that simulates all step propagation on a sample
one_sample = Function('one_sample',{states, controls, params}, {X});
%
% speedup trick: expand into scalar operations
one_sample = one_sample.expand();

%%%%%%%%%%%% Simulating the system %%%%%%%%%%

all_samples = one_sample.mapaccum('all_samples', N);

%%%%%%%%%%%% Identifying the simulated system %%%%%%%%%%
opts = struct;
% opts.ipopt.max_iter = 15;
% opts.ipopt.print_level = 3;%0,3
% opts.print_time = 1;
opts.ipopt.acceptable_tol = 1e-4;
opts.ipopt.acceptable_obj_change_tol = 1e-4;

single_multiple = 0;
switch single_multiple
    case 1           
        %%%%%%%%%%%% single shooting strategy %%%%%%%%%%
        X_symbolic = all_samples(x0, u_data, repmat(params,1,N));
        
        e = y_data-X_symbolic(1,:)';
        
        J = 1/N*dot(e,e);
        nlp = struct('x', params, 'f', J);
        
        solver = nlpsol('solver', 'ipopt', nlp, opts);
        
        sol = solver('x0', param_guess, 'lbx', lbx,'ubx', ubx);
        
        % parametros identificados:
        paramhat = sol.x.full;
    otherwise
        %%%%%%%%%%%% multiple shooting strategy %%%%%%%%%%
        % % All states become decision variables
        X = MX.sym('X', 2, N);

        res = one_sample.map(N, 'thread', 4);
        Xn = res(X, u_data', repmat(params,1,N));

        gaps = Xn(:,1:end-1)-X(:,2:end);

        e = y_data-Xn(1,:)';

        V = veccat(params, X);

        J = 1/N*dot(e,e);

        nlp = struct('x',V, 'f',J,'g',vec(gaps));

        % Multipleshooting allows for careful initialization
        yd = diff(y_data)*fs;
        X_guess = [ y_data  [yd;yd(end)]]';

        param_guess = [param_guess(:);X_guess(:)];

        solver = nlpsol('solver','ipopt', nlp, opts);

        sol = solver('x0',param_guess,'lbg',0,'ubg',0);
        solx = sol.x.full;
        paramhat = solx(1:nparam);
end

%% analisa resultado

Ihat    = denorm(paramhat(1),parammax(1),parammin(1));
khat   = denorm(paramhat(2),parammax(2),parammin(2));
bhat   = denorm(paramhat(3),parammax(3),parammin(3));
Fchat   = denorm(paramhat(4),parammax(4),parammin(4));

disp('Parametros identificados:')
[Ihat, khat, bhat, Fchat]


%% compare (real vs. CASADI) - train

Xhat = all_samples(x0, u_data, repmat(paramhat,1,N));
Xhat = Xhat.full;
yhat = Xhat(1,:)';

figure
hold on
plot(t,y_data,'k-','linewidth',1.5)
plot(t,yhat,'r--','linewidth',1.5)
grid on
xlabel('time')
legend({'real','casadi'},'location','best')
title('Train')

figure
hold on
plot(t,y_data-yhat,'r-','linewidth',1.5)
grid on
xlabel('time')
ylabel('error - train')
legend({'casadi'},'location','best')
title('Error - train')

%% output data
save outputFriction.mat yhat u_data y_data t

%% Dados de teste 

N_test = length(u_test);
t_test = ((1:N_test)-1) * Ts;

% Compare (real x CASADI) - test
Xhat_test = all_samples(x0_test, u_test, repmat(paramhat, 1, N_test));
Xhat_test = Xhat_test.full;
yhat_test = Xhat_test(1,:)';

figure
hold on
plot(t_test, y_test, 'b-', 'linewidth', 1.5) % Dados MF
plot(t_test, yhat_test, 'r--', 'linewidth', 1.5) % Saída Casadi
grid on
xlabel('Time')
legend({'Real MF', 'Casadi'}, 'location', 'best')
title('Test')

% Erro
figure
hold on
plot(t_test, y_test - yhat_test, 'm-', 'linewidth', 1.5)
grid on
xlabel('Time')
ylabel('Error')
legend({'Error (casadi)'}, 'location', 'best')
title('Error - test')

%% output data (test)
save outputTestFriction.mat yhat_test u_test y_test t_test

%%Métricas 

mse_train = mean((y_data - yhat).^2);
mse_test = mean((y_test - yhat_test).^2);

R_train = corrcoef(y_data, yhat);
R2_train = R_train(1,2)^2;

R_test = corrcoef(y_test, yhat_test);
R2_test = R_test(1,2)^2;

disp('-------------- Train x Test (Metrics) --------------')
fprintf('R² Train: %.4f\n', R2_train);
fprintf('MSE Train: %.4f\n', mse_train);
fprintf('R² Test: %.4f\n', R2_test);
fprintf('MSE Test: %.4f\n', mse_test);

disp('Parametros normalizados:')
disp(paramhat)
