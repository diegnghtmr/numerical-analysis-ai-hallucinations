% interpolacion_polinomica.m
%
% Este script implementa la interpolación polinómica de los datos
% experimentales de tasas de alucinación de modelos de lenguaje de
% distintos tamaños.  A partir de las tasas observadas para modelos de
% 7B, 13B, 30B (aproximado por 34B) y 70B parámetros, se construyen
% polinomios de interpolación mediante las fórmulas de Lagrange y de
% Newton por diferencias divididas, después se evalúan sobre una malla
% densa y en el punto de interés (40B parámetros).  Se compara la
% estimación con la interpolación pchip y se calculan errores de
% validación cruzada.  Los resultados se exportan a archivos CSV y
% figuras para ser incluidos en el informe.

function interpolacion_polinomica()
    % -----------------------------
    % 1. Definición de datos
    % -----------------------------
    % Números de parámetros de los modelos en número absoluto
    params = [7e9, 13e9, 30e9, 70e9];
    % Tasas de alucinación observadas (proporciones, no porcentajes)
    % Valores obtenidos de la tabla de la HEM/HHEM: Llama 2 7B (5,6 %),
    % Llama 2 13B (5,9 %), Llama 2 34B (3,3 %) y Llama 2 70B (5,1 %).
    % Para el nodo de 30B se utiliza la tasa de Llama 2 34B reportada en
    % la Tabla 6 de la encuesta de alucinaciones【538531971945421†L3682-L3693】,
    % calculada como complementaria de la precisión (96,70 %).
    rates = [0.056, 0.059, 0.033, 0.051];

    % Transformación a escala logarítmica para mejorar el condicionamiento
    X = log10(params);
    Y = rates;
    % Punto de interés: 40 mil millones de parámetros
    params_query = 40e9;
    xq = log10(params_query);

    % -----------------------------
    % 2. Construcción de polinomios
    % -----------------------------
    % Evaluación de Lagrange directamente en la malla de consulta
    xgrid = linspace(min(X), max(X), 200);
    y_lagrange_grid = lagrange_eval(X, Y, xgrid);
    y_lagrange_q = lagrange_eval(X, Y, xq);

    % Cálculo de coeficientes de Newton (diferencias divididas) y evaluación
    coeff_newton = newton_divided_diff(X, Y);
    y_newton_grid = newton_dd_eval(X, coeff_newton, xgrid);
    y_newton_q = newton_dd_eval(X, coeff_newton, xq);

    % Interpolación de referencia con PCHIP
    y_pchip_grid = interp1(X, Y, xgrid, 'pchip');
    y_pchip_q = interp1(X, Y, xq, 'pchip');

    % -----------------------------
    % 3. Tablas de salida
    % -----------------------------
    % Tabla de nodos
    tbl_nodos = table([7;13;30;70], params', X', Y', ...
        'VariableNames', {'ModelSize_B','Params','log10_Params','Rate'});
    writetable(tbl_nodos, 'tabla_nodos.csv');

    % Tabla de coeficientes (Newton)
    % Los coeficientes corresponden a la forma de Newton:
    % p(x) = c0 + c1*(x-x0) + c2*(x-x0)*(x-x1) + ...
    nCoeff = length(coeff_newton);
    var_names = cellstr(strcat('c', string(0:nCoeff-1)))';
    tbl_coef = array2table(coeff_newton(:)', 'VariableNames', var_names);
    writetable(tbl_coef, 'tabla_coeficientes_newton.csv');

    % Tabla de evaluación en la malla
    tbl_eval = table(xgrid', 10.^xgrid', y_lagrange_grid', y_newton_grid', y_pchip_grid', ...
        'VariableNames', {'log10_params','params','y_lagrange','y_newton','y_pchip'});
    writetable(tbl_eval, 'tabla_evaluacion_grid.csv');

    % -----------------------------
    % 4. Validación cruzada leave-one-out
    % -----------------------------
    n = length(X);
    idx = (1:n)';
    loocv_results = zeros(n, 9);
    % Columns: idx_removed, x_removed, y_true, y_lagr_pred, y_newt_pred, y_pchip_pred, err_lagr, err_newt, err_pchip
    for i = 1:n
        % Conjunto de entrenamiento (todas menos i)
        X_train = X([1:i-1,i+1:end]);
        Y_train = Y([1:i-1,i+1:end]);
        x_removed = X(i);
        y_true = Y(i);
        % Lagrange con nodos reducidos
        y_lagr_pred = lagrange_eval(X_train, Y_train, x_removed);
        % Newton con nodos reducidos
        coeff_n = newton_divided_diff(X_train, Y_train);
        y_newt_pred = newton_dd_eval(X_train, coeff_n, x_removed);
        % pchip con nodos reducidos
        y_pchip_pred = interp1(X_train, Y_train, x_removed, 'pchip');
        % Errores absolutos
        err_lagr = abs(y_true - y_lagr_pred);
        err_newt = abs(y_true - y_newt_pred);
        err_pchip = abs(y_true - y_pchip_pred);
        loocv_results(i,:) = [i, x_removed, y_true, y_lagr_pred, y_newt_pred, y_pchip_pred, err_lagr, err_newt, err_pchip];
    end
    % Convertir a tabla
    tbl_loocv = array2table(loocv_results, 'VariableNames', ...
        {'index_removed','log10_removed','rate_true','rate_lagr_pred','rate_newt_pred','rate_pchip_pred',...
        'error_lagr','error_newton','error_pchip'});
    writetable(tbl_loocv, 'tabla_loocv.csv');

    % -----------------------------
    % 5. Gráficas
    % -----------------------------
    % Figura 1: Ajuste de datos y polinomios
    figure;
    plot(10.^xgrid/1e9, y_lagrange_grid, 'b-', 'LineWidth',1.5, 'DisplayName','Lagrange'); hold on;
    plot(10.^xgrid/1e9, y_newton_grid, 'r--', 'LineWidth',1.5, 'DisplayName','Newton');
    plot(10.^xgrid/1e9, y_pchip_grid, 'g-.', 'LineWidth',1.5, 'DisplayName','PCHIP');
    % Puntos de datos
    scatter(params/1e9, Y, 60, 'k', 'filled', 'DisplayName','Datos');
    % Estimación en 40B
    scatter(params_query/1e9, y_lagrange_q, 70, 'b', 'o', 'filled', 'DisplayName','Estimación Lagrange 40B');
    scatter(params_query/1e9, y_newton_q, 70, 'r', 'o', 'filled', 'DisplayName','Estimación Newton 40B');
    scatter(params_query/1e9, y_pchip_q, 70, 'g', 'o', 'filled', 'DisplayName','Estimación PCHIP 40B');
    xlabel('Número de parámetros (en miles de millones)');
    ylabel('Tasa de alucinación');
    title('Interpolación polinómica de tasas de alucinación');
    legend('Location','best');
    grid on;
    saveas(gcf, 'fig_interpolacion_tasas.png');
    close;

    % Figura 2: Residuales en nodos
    figure;
    % Predicciones en nodos para cada método
    y_lagr_nodes = lagrange_eval(X, Y, X);
    y_newt_nodes = newton_dd_eval(X, coeff_newton, X);
    y_pchip_nodes = interp1(X, Y, X, 'pchip');
    bar([Y - y_lagr_nodes; Y - y_newt_nodes; Y - y_pchip_nodes]');
    set(gca,'XTickLabel',{'7B','13B','30B','70B'});
    xlabel('Nodo');
    ylabel('Residual (observado - interpolado)');
    legend({'Lagrange','Newton','PCHIP'}, 'Location','best');
    title('Residuos de interpolación en los nodos');
    grid on;
    saveas(gcf, 'fig_residuals.png');
    close;

    % Figura 3: Errores LOOCV
    figure;
    bar(tbl_loocv.index_removed, [tbl_loocv.error_lagr, tbl_loocv.error_newton, tbl_loocv.error_pchip]);
    set(gca,'XTick',1:n,'XTickLabel',{'7B','13B','30B','70B'});
    xlabel('Nodo eliminado');
    ylabel('Error absoluto');
    legend({'Lagrange','Newton','PCHIP'}, 'Location','best');
    title('Errores de validación cruzada leave-one-out');
    grid on;
    saveas(gcf, 'fig_loocv_errors.png');
    close;

    % Mostrar estimaciones
    fprintf('Estimación Lagrange en 40B: %.4f (%.2f%%)\n', y_lagrange_q, y_lagrange_q*100);
    fprintf('Estimación Newton en 40B:   %.4f (%.2f%%)\n', y_newton_q, y_newton_q*100);
    fprintf('Estimación PCHIP en 40B:    %.4f (%.2f%%)\n', y_pchip_q, y_pchip_q*100);
end

function yq = lagrange_eval(X, Y, xq)
    % Evaluación del polinomio de Lagrange en los puntos xq
    n = length(X);
    % Preasignar salida
    yq = zeros(size(xq));
    for j = 1:n
        % Cálculo del polinomio básico L_j(xq)
        L = ones(size(xq));
        for k = [1:j-1, j+1:n]
            L = L .* (xq - X(k)) / (X(j) - X(k));
        end
        yq = yq + Y(j) * L;
    end
end

function coeff = newton_divided_diff(X, Y)
    % Calcula los coeficientes de Newton mediante diferencias divididas
    n = length(X);
    % Inicializar una matriz triangular para almacenar las DD
    DD = zeros(n, n);
    DD(:,1) = Y(:);
    for j = 2:n
        for i = 1:n-j+1
            DD(i,j) = (DD(i+1,j-1) - DD(i,j-1)) / (X(i+j-1) - X(i));
        end
    end
    coeff = DD(1,:);
end

function yq = newton_dd_eval(X, coeff, xq)
    % Evalúa el polinomio de Newton dado un vector de coeficientes de
    % diferencias divididas.  X contiene los nodos de interpolación.
    n = length(coeff);
    yq = coeff(n) * ones(size(xq));
    for k = n-1:-1:1
        yq = yq .* (xq - X(k)) + coeff(k);
    end
end