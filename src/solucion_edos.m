% SOLUCION_EDOS.M
%
% Este script implementa los métodos de Euler explícito y Runge–Kutta de
% cuarto orden (RK4) para resolver numéricamente una ecuación
% diferencial ordinaria (EDO) que modela la evolución temporal de la
% confianza en un detector de alucinaciones. La EDO tiene la forma
%
%   dC/dt = -alpha * C + beta * Context(t),    C(0) = C0
%
% donde C(t) representa el nivel de confianza en que la respuesta de
% un modelo no es alucinatoria, alpha (>0) es el coeficiente de
% decaimiento (pérdida de confianza) y beta (>0) es el coeficiente de
% refuerzo debido al contexto. La función Context(t) puede provenir de
% datos experimentales (por ejemplo, la probabilidad de hallazgo de
% alucinaciones a lo largo del tiempo) o generarse sintéticamente. En
% ausencia de un conjunto de datos de contexto real, aquí se genera
% una señal compuesta por dos pulsos gausianos y una componente
% sinusoidal para simular las señales de detección.
%
% El script calcula soluciones aproximadas con los métodos de Euler y
% RK4 para distintos tamaños de paso h y las compara con una
% solución de referencia obtenida mediante RK4 con un paso muy fino
% h_ref. Se calculan el error máximo y el error al final del intervalo
% para cada método y se exportan los resultados en archivos CSV. Además
% se generan figuras que ilustran las trayectorias de C(t), el contexto
% y la convergencia del error con respecto al tamaño de paso.
%

% autor: [Diego Flores y Juan Mora]
% fecha: 20‑oct‑2025


clear; clc;

%% Parámetros del modelo y del experimento
alpha = 0.5;            % coeficiente de decaimiento (>0)
beta  = 1.0;            % coeficiente de refuerzo del contexto
C0    = 0.3;            % condición inicial de la confianza
Tmax  = 10.0;           % tiempo total de simulación

% Paso de referencia para la solución "exacta" (muy pequeño)
h_ref = 1e-3;

% Tamaños de paso a evaluar para los métodos de Euler y RK4
h_values = [1.0, 0.5, 0.25, 0.125, 0.0625];

%% Definición de la función de contexto
% La función Context(t) simula un patrón de señales que podrían
% representar instantes en los que el sistema de detección percibe
% indicios de alucinación. Se compone de dos pulsos gausianos y una
% oscilación senoidal suave más un nivel de ruido basal. Las funciones
% gausianas tienen centros en t=3 y t=7 con amplitudes diferentes.
context_function = @(t) 0.4*exp(-((t-3)/0.5).^2) + ...
                         0.3*exp(-((t-7)/0.7).^2) + ...
                         0.1*sin(2*pi*0.2*t) + ...
                         0.1;

% Definición del término derecho de la EDO: f(t, C) = -alpha*C + beta*Context(t)
f = @(t, C) -alpha*C + beta*context_function(t);

%% Solución de referencia mediante RK4 con paso h_ref
t_ref = 0:h_ref:Tmax;
N_ref = numel(t_ref);
C_ref = zeros(1, N_ref);
C_ref(1) = C0;
for i = 1:(N_ref-1)
    ti = t_ref(i);
    Ci = C_ref(i);
    % Cálculo de las pendientes para RK4
    k1 = f(ti, Ci);
    k2 = f(ti + h_ref/2, Ci + (h_ref/2)*k1);
    k3 = f(ti + h_ref/2, Ci + (h_ref/2)*k2);
    k4 = f(ti + h_ref,   Ci + h_ref*k3);
    % Actualizar C
    C_ref(i+1) = Ci + (h_ref/6)*(k1 + 2*k2 + 2*k3 + k4);
end

%% Inicializar tablas para almacenar errores
tabla_errores = [];
tabla_detalle  = [];

%% Bucle sobre cada tamaño de paso
for h = h_values
    % Crear malla temporal para este h
    t = 0:h:Tmax;
    N  = numel(t);
    % Evaluar contexto en los nodos de tiempo
    context_vals = context_function(t);

    % -----------------------------
    % Método de Euler explícito
    % -----------------------------
    C_euler = zeros(1, N);
    C_euler(1) = C0;
    for i = 1:(N-1)
        C_euler(i+1) = C_euler(i) + h * f(t(i), C_euler(i));
    end

    % -----------------------------
    % Método de Runge–Kutta de cuarto orden (RK4)
    % -----------------------------
    C_rk4 = zeros(1, N);
    C_rk4(1) = C0;
    for i = 1:(N-1)
        ti = t(i);
        Ci = C_rk4(i);
        % pendientes
        k1 = f(ti, Ci);
        k2 = f(ti + h/2, Ci + (h/2)*k1);
        k3 = f(ti + h/2, Ci + (h/2)*k2);
        k4 = f(ti + h,   Ci + h*k3);
        C_rk4(i+1) = Ci + (h/6)*(k1 + 2*k2 + 2*k3 + k4);
    end

    % -----------------------------
    % Comparación con la solución de referencia
    % -----------------------------
    % Interpolar la solución de referencia en los tiempos de este método
    C_ref_interp = interp1(t_ref, C_ref, t, 'pchip');
    % Calcular error máximo y error al final
    err_euler_max = max(abs(C_euler - C_ref_interp));
    err_euler_end = abs(C_euler(end) - C_ref_interp(end));
    err_rk4_max   = max(abs(C_rk4 - C_ref_interp));
    err_rk4_end   = abs(C_rk4(end) - C_ref_interp(end));

    % Guardar resultados en tablas
    tabla_errores = [tabla_errores; h, N, err_euler_max, err_euler_end, err_rk4_max, err_rk4_end];

    % Guardar detalle del último tiempo para referencia (h, método, valor)
    tabla_detalle = [tabla_detalle; h, C_euler(end), C_rk4(end), C_ref_interp(end)];

    % Exportar curvas de C(t) para este h en archivos separados
    curva_datos = [t(:), context_vals(:), C_euler(:), C_rk4(:), C_ref_interp(:)];
    fname_curva = sprintf('curva_EDO_h_%g.csv', h);
    writecell([{"t", "Context", "Euler", "RK4", "Ref"}; num2cell(curva_datos)], fname_curva);
end

%% Exportar tabla de errores a CSV
% Columnas: [h, N, err_euler_max, err_euler_end, err_rk4_max, err_rk4_end]
tabla_header = {"h", "N", "ErrorEulerMax", "ErrorEulerFin", "ErrorRK4Max", "ErrorRK4Fin"};
writecell([tabla_header; num2cell(tabla_errores)], 'tabla_errores_EDO.csv');

%% Exportar tabla de detalle de valores finales
detalle_header = {"h", "C_euler_fin", "C_rk4_fin", "C_ref_fin"};
writecell([detalle_header; num2cell(tabla_detalle)], 'tabla_detalle_EDO.csv');

%% Dibujar gráficas
% 1. Solución C(t) para el paso más grande y más pequeño
h_show = [h_values(1), h_values(end)];
fig1 = figure('Name', 'C(t) y Contexto para distintos h', 'NumberTitle', 'off', 'Position', [100, 100, 1100, 600]);
hold on;
plot(t_ref, C_ref, '-', 'LineWidth', 2.5, 'Color', [0.49 0.18 0.56], 'DisplayName', 'Referencia (RK4 fino)');
colores = [0.00 0.45 0.74; 0.85 0.33 0.10];
for j = 1:numel(h_show)
    h = h_show(j);
    % cargar curva desde CSV usando readtable para manejar encabezados
    fname_curva = sprintf('curva_EDO_h_%g.csv', h);
    data_table = readtable(fname_curva);
    t_tmp  = data_table.t;
    C_e    = data_table.Euler;
    C_r    = data_table.RK4;
    plot(t_tmp, C_e, 'o--', 'Color', colores(j,:), 'LineWidth', 1.8, 'MarkerSize', 5, ...
         'DisplayName', sprintf('Euler (h=%.3f)', h));
    plot(t_tmp, C_r, 's-',  'Color', colores(j,:), 'LineWidth', 2.2, 'MarkerSize', 4, ...
         'DisplayName', sprintf('RK4 (h=%.3f)', h));
end
% Graficar contexto en escala secundaria
yyaxis right;
plot(t_ref, context_function(t_ref), ':', 'Color', [0.5 0.5 0.5], 'LineWidth', 2, 'DisplayName', 'Contexto');
ylabel('Señal de contexto', 'FontSize', 13);
set(gca, 'YColor', [0.5 0.5 0.5]);
yyaxis left;
ylabel('Confianza C(t)', 'FontSize', 13);
xlabel('Tiempo t', 'FontSize', 13);
title('Evolución de la confianza C(t) con distintos métodos y tamaños de paso', 'FontSize', 16, 'FontWeight', 'bold');
legend('Location', 'northeastoutside', 'FontSize', 10);
grid on;
set(gca, 'FontSize', 12, 'GridAlpha', 0.3, 'LineWidth', 0.8);
box on;
print('fig_EDO_soluciones.png', '-dpng', '-r300');

% 2. Error máximo vs h en escala log-log
fig2 = figure('Name', 'Error máximo vs h', 'NumberTitle', 'off', 'Position', [100, 100, 1000, 600]);
loglog(tabla_errores(:,1), tabla_errores(:,3), 'o-', 'LineWidth', 2.5, 'MarkerSize', 8, ...
       'MarkerFaceColor', [0.00 0.45 0.74], 'Color', [0.00 0.45 0.74], 'DisplayName', 'Euler');
hold on;
loglog(tabla_errores(:,1), tabla_errores(:,5), 's-', 'LineWidth', 2.5, 'MarkerSize', 8, ...
       'MarkerFaceColor', [0.85 0.33 0.10], 'Color', [0.85 0.33 0.10], 'DisplayName', 'RK4');
grid on;
xlabel('Tamaño de paso (h)', 'FontSize', 13);
ylabel('Error máximo (escala log)', 'FontSize', 13);
title('Convergencia del error máximo: Euler vs RK4', 'FontSize', 16, 'FontWeight', 'bold');
legend('Location', 'northwest', 'FontSize', 11);
set(gca, 'FontSize', 12, 'GridAlpha', 0.3, 'LineWidth', 0.8);
box on;
print('fig_EDO_error_maximo.png', '-dpng', '-r300');

% 3. Error al final vs h en escala log-log
fig3 = figure('Name', 'Error al final vs h', 'NumberTitle', 'off', 'Position', [100, 100, 1000, 600]);
loglog(tabla_errores(:,1), tabla_errores(:,4), 'o-', 'LineWidth', 2.5, 'MarkerSize', 8, ...
       'MarkerFaceColor', [0.00 0.45 0.74], 'Color', [0.00 0.45 0.74], 'DisplayName', 'Euler');
hold on;
loglog(tabla_errores(:,1), tabla_errores(:,6), 's-', 'LineWidth', 2.5, 'MarkerSize', 8, ...
       'MarkerFaceColor', [0.85 0.33 0.10], 'Color', [0.85 0.33 0.10], 'DisplayName', 'RK4');
grid on;
xlabel('Tamaño de paso (h)', 'FontSize', 13);
ylabel('Error al final del intervalo (escala log)', 'FontSize', 13);
title('Convergencia del error al final: Euler vs RK4', 'FontSize', 16, 'FontWeight', 'bold');
legend('Location', 'northwest', 'FontSize', 11);
set(gca, 'FontSize', 12, 'GridAlpha', 0.3, 'LineWidth', 0.8);
box on;
print('fig_EDO_error_final.png', '-dpng', '-r300');

%% Mostrar resultados en consola
disp('Tabla de errores (h, N, error max Euler, error final Euler, error max RK4, error final RK4):');
disp(tabla_errores);