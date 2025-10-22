% COMPARACION_PRECISION_COSTOS.M
%
% Este guion compara la precisión y el costo computacional de los
% principales métodos numéricos implementados en el proyecto sobre
% alucinaciones de modelos de lenguaje. La comparación abarca las
% aproximaciones por series de Taylor, los métodos de búsqueda de
% raíces (Bisección y Newton), la interpolación polinómica (ajuste
% polinómico), la integración numérica (regla del trapecio y regla de
% Simpson), los modelos de regresión (lineal y cuadrático) y la
% resolución de EDOs (Euler y Runge–Kutta de cuarto orden).  Para cada
% método se calcula una métrica de error representativa y se estima el
% costo computacional mediante `tic`/`toc`, número de evaluaciones de
% funciones y número de iteraciones o pasos.  Los resultados se
% resumen en un cuadro comparativo y se guardan en un archivo CSV.
%
% Notas:
% - Este script está diseñado como plantilla.  Muchos métodos usan
%   datos sintéticos (por ejemplo, señales de contexto o pares FPR/TPR) y
%   no pretende reemplazar los análisis detallados de cada sección.
% - Para obtener tiempos reproducibles es recomendable cerrar otros
%   procesos y ejecutar varias repeticiones usando `timeit`.
% - Los resultados dependerán del hardware y versión de MATLAB.
%
% Autor: [Nombre del autor]
% Fecha: [Fecha de creación]

clear; clc;

% Inicializar contenedores de resultados
metodos   = {};
errores   = [];
costos    = [];
evaluaciones = [];
notas     = {};

%% 1. Aproximaciones por series de Taylor
% Se comparan las aproximaciones de la sigmoide y la tangente hiperbólica
% usando series de Taylor de órdenes 3, 4 y 5 en el intervalo [-8,8].
x = linspace(-8, 8, 2001);
y_sig = 1./(1+exp(-x)); % función sigmoide exacta
y_tanh = tanh(x);        % función tanh exacta
ordenes = [3 4 5];
for N = ordenes
    % Sigmoide
    tic;
    y_sig_approx = taylor_sigmoid(x, N);
    tiempo = toc;
    err_max = max(abs(y_sig_approx - y_sig));
    rmse = sqrt(mean((y_sig_approx - y_sig).^2));
    % Número de operaciones ~ evaluaciones de términos de la serie
    num_ops = numel(x)*(N+1);
    metodos{end+1} = sprintf('Taylor sigmoide N=%d', N);
    errores(end+1,:) = [err_max, rmse];
    costos(end+1) = tiempo;
    evaluaciones(end+1) = num_ops;
    notas{end+1} = 'Aprox. de la sigmoide con serie de Taylor';
    % Tanh
    tic;
    y_tanh_approx = taylor_tanh(x, N);
    tiempo = toc;
    err_max = max(abs(y_tanh_approx - y_tanh));
    rmse = sqrt(mean((y_tanh_approx - y_tanh).^2));
    num_ops = numel(x)*(N+1);
    metodos{end+1} = sprintf('Taylor tanh N=%d', N);
    errores(end+1,:) = [err_max, rmse];
    costos(end+1) = tiempo;
    evaluaciones(end+1) = num_ops;
    notas{end+1} = 'Aprox. de tanh con serie de Taylor';
end

%% 2. Métodos de búsqueda de raíces (función de prueba)
% Se define una función sencilla con raíz conocida (por ejemplo f(t)=tanh(t))
% para comparar la eficiencia de Bisección y Newton.  La raíz está en t=0.
f_root = @(t) tanh(t);
df_root = @(t) sech(t).^2; % derivada exacta para Newton
a0 = -5; b0 = 5; tol = 1e-6; maxit = 100;

% Bisección
tic;
[root_bis, n_bis] = biseccion(f_root, a0, b0, tol, maxit);
tiempo_bis = toc;
err_bis = abs(root_bis - 0);
evals_bis = 2*n_bis; % dos evaluaciones de f por iteración
metodos{end+1} = 'Búsqueda de raíces: Bisección';
errores(end+1,:) = [err_bis, err_bis];
costos(end+1) = tiempo_bis;
evaluaciones(end+1) = evals_bis;
notas{end+1} = sprintf('%.0f iteraciones', n_bis);

% Newton-Raphson
x0 = 2; % valor inicial arbitrario
tic;
[root_newt, n_newt] = newton_method(f_root, df_root, x0, tol, maxit);
tiempo_newt = toc;
err_newt = abs(root_newt - 0);
evals_newt = 2*n_newt; % f y df por iteración
metodos{end+1} = 'Búsqueda de raíces: Newton';
errores(end+1,:) = [err_newt, err_newt];
costos(end+1) = tiempo_newt;
evaluaciones(end+1) = evals_newt;
notas{end+1} = sprintf('%.0f iteraciones', n_newt);

%% 3. Interpolación polinómica (ajuste polinómico de grado 3)
% Datos de alucinación para modelos Llama-2 (frecuencias obtenidas en
% secciones anteriores).
params = [7e9, 13e9, 30e9, 70e9];
y_hallu = [0.056, 0.059, 0.033, 0.051];
x_nodes = log10(params);
% Ajuste de grado 3 (equivalente a Lagrange/Newton con 4 puntos)
tic;
p3 = polyfit(x_nodes, y_hallu, 3);
tiempo_fit = toc;
% Evaluación en nodos
tic;
y_fit = polyval(p3, x_nodes);
tiempo_eval = toc;
rmse_fit = sqrt(mean((y_fit - y_hallu).^2));
% LOOCV: error promedio al dejar uno fuera
err_loocv = 0;
for i = 1:numel(x_nodes)
    idx = (1:numel(x_nodes)) ~= i;
    pp = polyfit(x_nodes(idx), y_hallu(idx), 3);
    yi_pred = polyval(pp, x_nodes(i));
    err_loocv = err_loocv + (yi_pred - y_hallu(i))^2;
end
rmse_loocv = sqrt(err_loocv/numel(x_nodes));
metodos{end+1} = 'Interpolación polinómica (grado 3)';
errores(end+1,:) = [rmse_fit, rmse_loocv];
costos(end+1) = tiempo_fit + tiempo_eval;
evaluaciones(end+1) = numel(x_nodes)^2; % coste aproximado O(n^2) para ajustar
notas{end+1} = 'Coeficientes mediante polyfit y LOOCV';

%% 4. Integración numérica (regla del trapecio vs Simpson)
% Se integra una función suave conocida para disponer de solución exacta.
g = @(x) exp(-x.^2);  % integral exacta sobre [0,1] ~ 0.746824
actual_int = 0.7468241328124271; % valor de referencia aproximado
% Generar malla irregular
xi = sort(rand(1,500));
yi = g(xi);
tic;
auc_trap = trapz(xi, yi);
tiempo_trap = toc;
% Remuestreo uniforme y Simpson compuesto
M = 1000;
xu = linspace(0,1,M);
yu = interp1(xi, yi, xu, 'pchip');
tic;
auc_simp = simpson_compuesto(xu, yu);
tiempo_simp = toc;
err_trap = abs(auc_trap - actual_int);
err_simp = abs(auc_simp - actual_int);
metodos{end+1} = 'Integración: Trapecio';
errores(end+1,:) = [err_trap, err_trap];
costos(end+1) = tiempo_trap;
evaluaciones(end+1) = numel(xi);
notas{end+1} = 'Malla irregular 500 pts.';
metodos{end+1} = 'Integración: Simpson';
errores(end+1,:) = [err_simp, err_simp];
costos(end+1) = tiempo_simp;
evaluaciones(end+1) = numel(xu);
notas{end+1} = sprintf('Remuestreo %d pts.', M);

%% 5. Modelos de regresión (lineal y cuadrático)
tic;
p1 = polyfit(x_nodes, y_hallu, 1);
tiempo_lin = toc;
y_pred1 = polyval(p1, x_nodes);
rmse1 = sqrt(mean((y_pred1 - y_hallu).^2));
ss_tot = sum((y_hallu - mean(y_hallu)).^2);
ss_res = sum((y_hallu - y_pred1).^2);
r2_1 = 1 - ss_res/ss_tot;
metodos{end+1} = 'Regresión lineal';
errores(end+1,:) = [rmse1, r2_1];
costos(end+1) = tiempo_lin;
evaluaciones(end+1) = numel(x_nodes);
notas{end+1} = 'Ajuste lineal en log10(params)';

tic;
p2 = polyfit(x_nodes, y_hallu, 2);
tiempo_quad = toc;
y_pred2 = polyval(p2, x_nodes);
rmse2 = sqrt(mean((y_pred2 - y_hallu).^2));
ss_res2 = sum((y_hallu - y_pred2).^2);
r2_2 = 1 - ss_res2/ss_tot;
metodos{end+1} = 'Regresión cuadrática';
errores(end+1,:) = [rmse2, r2_2];
costos(end+1) = tiempo_quad;
evaluaciones(end+1) = numel(x_nodes);
notas{end+1} = 'Ajuste cuadrático en log10(params)';

%% 6. Solución de EDOs (Euler y RK4)
alpha = 0.5; beta = 1.0; C0 = 0.3; Tmax = 10;
context_fun = @(t) 0.4*exp(-((t-3)/0.5).^2) + 0.3*exp(-((t-7)/0.7).^2) + 0.1*sin(2*pi*0.2*t) + 0.1;
f_edo = @(t,C) -alpha*C + beta*context_fun(t);
% Solución de referencia con RK4 y paso muy fino
h_ref = 1e-3; t_ref = 0:h_ref:Tmax; C_ref = rk4_solver(f_edo, C0, t_ref);
% Tamaño de paso para comparación
h_e = 0.125; t_e = 0:h_e:Tmax;
% Euler
tic;
C_e = euler_solver(f_edo, C0, t_e);
tiempo_euler = toc;
C_ref_interp = interp1(t_ref, C_ref, t_e, 'pchip');
err_e_max = max(abs(C_e - C_ref_interp));
metodos{end+1} = 'EDO: Euler';
errores(end+1,:) = [err_e_max, err_e_max];
costos(end+1) = tiempo_euler;
evaluaciones(end+1) = numel(t_e) - 1; % una eval por paso
notas{end+1} = sprintf('h=%.3f', h_e);
% RK4
tic;
C_r = rk4_solver(f_edo, C0, t_e);
tiempo_rk4 = toc;
err_r_max = max(abs(C_r - C_ref_interp));
metodos{end+1} = 'EDO: RK4';
errores(end+1,:) = [err_r_max, err_r_max];
costos(end+1) = tiempo_rk4;
evaluaciones(end+1) = 4*(numel(t_e) - 1); % cuatro evals por paso
notas{end+1} = sprintf('h=%.3f', h_e);

%% Construir tabla final
num_metodos = numel(metodos);
tabla = cell(num_metodos+1, 6);
tabla(1,:) = {'Método', 'Error_1 (p.ej. L∞/RMSE)', 'Error_2 (sec.)', 'Tiempo (s)', '#Eval f', 'Notas'};
for i = 1:num_metodos
    tabla{i+1,1} = metodos{i};
    tabla{i+1,2} = errores(i,1);
    tabla{i+1,3} = errores(i,2);
    tabla{i+1,4} = costos(i);
    tabla{i+1,5} = evaluaciones(i);
    tabla{i+1,6} = notas{i};
end

% Exportar tabla a CSV
writetable(cell2table(tabla(2:end,:), 'VariableNames', tabla(1,:)), 'matriz_comparacion.csv');

% Mostrar resumen en pantalla
disp('Resumen comparativo de precisión y costo computacional:');
disp(tabla);

%% --- FUNCIONES AUXILIARES ---
function y = taylor_sigmoid(x, N)
    % Aproximación de la función sigmoide mediante serie de Taylor en 0
    % f(x) = 1/2 + x/4 - x^3/48 + ...
    % Derivadas de la sigmoide en 0 pueden calcularse simbólicamente, pero aquí
    % se usa el desarrollo manual hasta orden N.
    y = 0.5 * ones(size(x));
    if N >= 1
        y = y + x/4;
    end
    if N >= 3
        y = y - x.^3/48;
    end
    if N >= 5
        y = y + x.^5/480;
    end
    if N >= 7
        y = y - x.^7/80640;
    end
    % Para órdenes mayores se pueden añadir términos siguiendo la derivación de
    % las derivadas en 0.
end

function y = taylor_tanh(x, N)
    % Aproximación de la tanh mediante serie de Taylor en 0
    % tanh(x) = x - x^3/3 + 2x^5/15 - 17x^7/315 + ...
    y = zeros(size(x));
    if N >= 1
        y = y + x;
    end
    if N >= 3
        y = y - x.^3/3;
    end
    if N >= 5
        y = y + 2*x.^5/15;
    end
    if N >= 7
        y = y - 17*x.^7/315;
    end
end

function [root, iter] = biseccion(f, a, b, tol, maxit)
    % Método de bisección para hallar una raíz de f en [a,b]
    fa = f(a); fb = f(b);
    if fa*fb > 0
        error('La función no cambia de signo en el intervalo.');
    end
    iter = 0;
    while (b - a)/2 > tol && iter < maxit
        c = (a + b)/2;
        fc = f(c);
        if fa*fc <= 0
            b = c; fb = fc;
        else
            a = c; fa = fc;
        end
        iter = iter + 1;
    end
    root = (a + b)/2;
end

function [root, iter] = newton_method(f, df, x0, tol, maxit)
    % Método de Newton-Raphson con derivada exacta
    x = x0;
    iter = 0;
    for k = 1:maxit
        fx = f(x);
        dfx = df(x);
        if abs(dfx) < eps
            break;
        end
        x_new = x - fx/dfx;
        if abs(x_new - x) < tol
            x = x_new;
            iter = k;
            break;
        end
        x = x_new;
    end
    root = x;
end

function integral = simpson_compuesto(x, y)
    % Regla de Simpson compuesta sobre malla uniforme x
    n = numel(x);
    if mod(n-1,2) ~= 0
        error('Simpson requiere número par de subintervalos.');
    end
    h = x(2) - x(1);
    integral = y(1) + y(end);
    integral = integral + 4*sum(y(2:2:end-1));
    integral = integral + 2*sum(y(3:2:end-2));
    integral = integral * h/3;
end

function C = euler_solver(f, C0, t)
    % Método de Euler explícito para resolver C'=f(t,C)
    C = zeros(size(t));
    C(1) = C0;
    h = t(2) - t(1);
    for i = 1:(numel(t)-1)
        C(i+1) = C(i) + h * f(t(i), C(i));
    end
end

function C = rk4_solver(f, C0, t)
    % Método de Runge-Kutta de cuarto orden para C'=f(t,C)
    C = zeros(size(t));
    C(1) = C0;
    h = t(2) - t(1);
    for i = 1:(numel(t)-1)
        ti = t(i);
        Ci = C(i);
        k1 = f(ti, Ci);
        k2 = f(ti + h/2, Ci + (h/2)*k1);
        k3 = f(ti + h/2, Ci + (h/2)*k2);
        k4 = f(ti + h,   Ci + h*k3);
        C(i+1) = Ci + (h/6)*(k1 + 2*k2 + 2*k3 + k4);
    end
end