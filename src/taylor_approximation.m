% taylor_approximation.m
%
% Este script implementa la aproximación de las funciones de activación
% sigmoid y tanh mediante series de Taylor centradas en x0=0. El objetivo
% es cuantificar el error de truncamiento para diferentes órdenes de
% aproximación y generar tablas y gráficos que sirvan como insumo para
% el informe de análisis numérico de alucinaciones.
%
% El script define las funciones vectorizadas `taylor_sigmoid` y
% `taylor_tanh`, que calculan el polinomio de Taylor de orden N para
% cada función. Estas funciones utilizan la toolbox simbólica para
% construir los polinomios y luego evalúan los resultados sobre un
% vector de entrada `x`.

% autor: [Diego Flores y Juan Mora]
% fecha: 20‑oct‑2025

function taylor_approximation()
    % Rango de valores de entrada (logits típicos)
    x = linspace(-8, 8, 1001)';
    % Órdenes de Taylor a evaluar (preferentemente impares para capturar
    % simetría y evitar la desaparición de términos pares en tanh)
    orders = [3, 5, 7, 9, 11];

    % Referencias exactas usando funciones incorporadas de MATLAB
    y_ref_sig = 1./(1+exp(-x));
    y_ref_tanh = tanh(x);

    % Inicializar tablas de resultados
    % Cada tabla contendrá columnas: x, y_ref, y_taylor, abs_err, rel_err, orden
    tabla_sig = [];
    tabla_tanh = [];
    % Métricas por orden
    metrics_sig = zeros(length(orders), 3); % columnas: N, maxError, RMSE
    metrics_tanh = zeros(length(orders), 3);

    % Iterar sobre cada orden y calcular aproximaciones y errores
    for idx = 1:length(orders)
        N = orders(idx);
        % Aproximaciones
        y_taylor_sig = taylor_sigmoid(x, N);
        y_taylor_tanh = taylor_tanh(x, N);

        % Calcular errores
        abs_err_sig = abs(y_ref_sig - y_taylor_sig);
        rel_err_sig = abs_err_sig ./ max(abs(y_ref_sig), eps);
        abs_err_tanh = abs(y_ref_tanh - y_taylor_tanh);
        rel_err_tanh = abs_err_tanh ./ max(abs(y_ref_tanh), eps);

        % Guardar métricas (orden, error infinito, RMSE)
        metrics_sig(idx, :) = [N, max(abs_err_sig), sqrt(mean(abs_err_sig.^2))];
        metrics_tanh(idx, :) = [N, max(abs_err_tanh), sqrt(mean(abs_err_tanh.^2))];

        % Construir tablas para cada orden y concatenar
        tabla_sig = [tabla_sig; [x, y_ref_sig, y_taylor_sig, abs_err_sig, rel_err_sig, repmat(N, numel(x), 1)]];
        tabla_tanh = [tabla_tanh; [x, y_ref_tanh, y_taylor_tanh, abs_err_tanh, rel_err_tanh, repmat(N, numel(x), 1)]];
    end

    % Exportar tablas a CSV
    sig_hdr = {'x','y_ref','y_taylor','abs_err','rel_err','orden'};
    tanh_hdr = sig_hdr;
    writetable(array2table(tabla_sig, 'VariableNames', sig_hdr), 'taylor_sigmoid_datos.csv');
    writetable(array2table(tabla_tanh, 'VariableNames', tanh_hdr), 'taylor_tanh_datos.csv');

    % Guardar métricas en archivo
    metrics_hdr = {'orden','max_err','RMSE'};
    writetable(array2table(metrics_sig, 'VariableNames', metrics_hdr), 'taylor_sigmoid_metricas.csv');
    writetable(array2table(metrics_tanh, 'VariableNames', metrics_hdr), 'taylor_tanh_metricas.csv');

    % Generar gráficas para sigmoid
    % Paleta de colores profesional (gradiente de azul a naranja)
    colors = [0.12 0.47 0.71; 0.20 0.63 0.79; 0.99 0.55 0.38; 0.89 0.10 0.11; 0.55 0.00 0.52];

    figure('Position', [100, 100, 1000, 600]);
    hold on;
    plot(x, y_ref_sig, 'k', 'LineWidth', 2.5, 'DisplayName', 'Referencia (exacta)');
    for idx = 1:length(orders)
        N = orders(idx);
        y_taylor_sig = tabla_sig(tabla_sig(:,6)==N, 3);
        plot(x, y_taylor_sig, 'LineWidth', 2.0, 'Color', colors(idx,:), ...
             'DisplayName', ['Orden N=', num2str(N)], 'LineStyle', '--');
    end
    title('Aproximación de sigmoid mediante series de Taylor', 'FontSize', 16, 'FontWeight', 'bold');
    xlabel('x (logits)', 'FontSize', 13);
    ylabel('sigmoid(x)', 'FontSize', 13);
    legend('Location', 'northwest', 'FontSize', 11);
    grid on;
    set(gca, 'FontSize', 12, 'GridAlpha', 0.3, 'LineWidth', 0.8);
    box on;
    hold off;
    print('fig_taylor_sigmoid_aprox.png', '-dpng', '-r300');
    close;

    % Error absoluto vs x para sigmoid (escala logarítmica en y)
    figure('Position', [100, 100, 1000, 600]);
    hold on;
    for idx = 1:length(orders)
        N = orders(idx);
        err_sig = tabla_sig(tabla_sig(:,6)==N, 4);
        semilogy(x, err_sig, 'LineWidth', 2.0, 'Color', colors(idx,:), ...
                 'DisplayName', ['Orden N=', num2str(N)]);
    end
    title('Error absoluto de la aproximación de sigmoid', 'FontSize', 16, 'FontWeight', 'bold');
    xlabel('x (logits)', 'FontSize', 13);
    ylabel('Error absoluto |error|', 'FontSize', 13);
    legend('Location', 'best', 'FontSize', 11);
    grid on;
    set(gca, 'FontSize', 12, 'GridAlpha', 0.3, 'LineWidth', 0.8);
    box on;
    hold off;
    print('fig_taylor_sigmoid_error.png', '-dpng', '-r300');
    close;

    % Métricas de error vs N para sigmoid
    figure('Position', [100, 100, 1000, 600]);
    semilogy(metrics_sig(:,1), metrics_sig(:,2), '-o', 'LineWidth', 2.5, ...
             'MarkerSize', 8, 'MarkerFaceColor', [0.85 0.33 0.10], ...
             'Color', [0.85 0.33 0.10], 'DisplayName','Error máximo');
    hold on;
    semilogy(metrics_sig(:,1), metrics_sig(:,3), '-s', 'LineWidth', 2.5, ...
             'MarkerSize', 8, 'MarkerFaceColor', [0.00 0.45 0.74], ...
             'Color', [0.00 0.45 0.74], 'DisplayName','RMSE');
    title('Convergencia del error: sigmoid', 'FontSize', 16, 'FontWeight', 'bold');
    xlabel('Orden de la serie de Taylor (N)', 'FontSize', 13);
    ylabel('Error (escala logarítmica)', 'FontSize', 13);
    legend('Location', 'northeast', 'FontSize', 11);
    grid on;
    set(gca, 'FontSize', 12, 'GridAlpha', 0.3, 'LineWidth', 0.8);
    box on;
    hold off;
    print('fig_taylor_sigmoid_metricas.png', '-dpng', '-r300');
    close;

    % Generar gráficas para tanh
    figure('Position', [100, 100, 1000, 600]);
    hold on;
    plot(x, y_ref_tanh, 'k', 'LineWidth', 2.5, 'DisplayName', 'Referencia (exacta)');
    for idx = 1:length(orders)
        N = orders(idx);
        y_taylor_tanh = tabla_tanh(tabla_tanh(:,6)==N, 3);
        plot(x, y_taylor_tanh, 'LineWidth', 2.0, 'Color', colors(idx,:), ...
             'DisplayName', ['Orden N=', num2str(N)], 'LineStyle', '--');
    end
    title('Aproximación de tanh mediante series de Taylor', 'FontSize', 16, 'FontWeight', 'bold');
    xlabel('x (logits)', 'FontSize', 13);
    ylabel('tanh(x)', 'FontSize', 13);
    legend('Location', 'northwest', 'FontSize', 11);
    grid on;
    set(gca, 'FontSize', 12, 'GridAlpha', 0.3, 'LineWidth', 0.8);
    box on;
    hold off;
    print('fig_taylor_tanh_aprox.png', '-dpng', '-r300');
    close;

    % Error absoluto vs x para tanh
    figure('Position', [100, 100, 1000, 600]);
    hold on;
    for idx = 1:length(orders)
        N = orders(idx);
        err_tanh = tabla_tanh(tabla_tanh(:,6)==N, 4);
        semilogy(x, err_tanh, 'LineWidth', 2.0, 'Color', colors(idx,:), ...
                 'DisplayName', ['Orden N=', num2str(N)]);
    end
    title('Error absoluto de la aproximación de tanh', 'FontSize', 16, 'FontWeight', 'bold');
    xlabel('x (logits)', 'FontSize', 13);
    ylabel('Error absoluto |error|', 'FontSize', 13);
    legend('Location', 'best', 'FontSize', 11);
    grid on;
    set(gca, 'FontSize', 12, 'GridAlpha', 0.3, 'LineWidth', 0.8);
    box on;
    hold off;
    print('fig_taylor_tanh_error.png', '-dpng', '-r300');
    close;

    % Métricas de error vs N para tanh
    figure('Position', [100, 100, 1000, 600]);
    semilogy(metrics_tanh(:,1), metrics_tanh(:,2), '-o', 'LineWidth', 2.5, ...
             'MarkerSize', 8, 'MarkerFaceColor', [0.85 0.33 0.10], ...
             'Color', [0.85 0.33 0.10], 'DisplayName','Error máximo');
    hold on;
    semilogy(metrics_tanh(:,1), metrics_tanh(:,3), '-s', 'LineWidth', 2.5, ...
             'MarkerSize', 8, 'MarkerFaceColor', [0.00 0.45 0.74], ...
             'Color', [0.00 0.45 0.74], 'DisplayName','RMSE');
    title('Convergencia del error: tanh', 'FontSize', 16, 'FontWeight', 'bold');
    xlabel('Orden de la serie de Taylor (N)', 'FontSize', 13);
    ylabel('Error (escala logarítmica)', 'FontSize', 13);
    legend('Location', 'northeast', 'FontSize', 11);
    grid on;
    set(gca, 'FontSize', 12, 'GridAlpha', 0.3, 'LineWidth', 0.8);
    box on;
    hold off;
    print('fig_taylor_tanh_metricas.png', '-dpng', '-r300');
    close;

    % Informar en la consola que la ejecución ha finalizado
    fprintf('La aproximación por series de Taylor se ha ejecutado correctamente.\n');
end

function y = taylor_sigmoid(x, N)
    % Aproxima la función sigmoid mediante un polinomio de Taylor de orden N
    % centrado en 0. Utiliza la toolbox simbólica para construir el
    % polinomio y evalúa en el vector x. Se almacena en caché para mejorar
    % rendimiento en llamadas repetidas.
    persistent poly_cache_sig;
    if isempty(poly_cache_sig)
        poly_cache_sig = containers.Map('KeyType','double','ValueType','any');
    end
    if isKey(poly_cache_sig, N)
        P = poly_cache_sig(N);
    else
        syms t;
        f = 1/(1+exp(-t));
        P = taylor(f, t, 'ExpansionPoint', 0, 'Order', N+1);
        poly_cache_sig(N) = P;
    end
    y = double(subs(P, t, x));
end

function y = taylor_tanh(x, N)
    % Aproxima la función tanh mediante un polinomio de Taylor de orden N
    % centrado en 0. Utiliza la toolbox simbólica para construir el
    % polinomio y evalúa en el vector x. Se almacena en caché para mejorar
    % rendimiento en llamadas repetidas.
    persistent poly_cache_tanh;
    if isempty(poly_cache_tanh)
        poly_cache_tanh = containers.Map('KeyType','double','ValueType','any');
    end
    if isKey(poly_cache_tanh, N)
        P = poly_cache_tanh(N);
    else
        syms t;
        f = tanh(t);
        P = taylor(f, t, 'ExpansionPoint', 0, 'Order', N+1);
        poly_cache_tanh(N) = P;
    end
    y = double(subs(P, t, x));
end
