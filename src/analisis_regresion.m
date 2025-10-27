%% analisis_regresion.m
% Este script implementa la sección 7 del proyecto: análisis de regresión
% para modelado predictivo de la tasa de alucinaciones en función de la
% complejidad del modelo, medida como el logaritmo decimal del número de
% parámetros (en miles de millones).  Se ajustan modelos de regresión
% lineal y cuadrática, se comparan sus desempeños mediante métricas de
% error y criterio de información, y se generan tablas y figuras
% relacionadas con los coeficientes, predicciones y residuales.

% Autores: [Diego Flores y Juan Mora]
% Fecha: 20‑oct‑2025


function analisis_regresion()
    %% 1. Definición de datos
    % La tabla incluye el nombre del modelo, el número de parámetros
    % (expresado en miles de millones) y la tasa de alucinación observada.
    % Estos valores se basan en reportes del HHEM v2 y otros estudios
    % recientes.  Si se dispone de un archivo CSV externo denominado
    % 'datos_regresion.csv' con columnas 'Model','Params','HallRate', el
    % script lo leerá; de lo contrario se utilizarán los valores
    % codificados.
    dataFile = 'datos_regresion.csv';
    if exist(dataFile, 'file')
        T = readtable(dataFile);
    else
        % Datos codificados: modelos representativos de la familia Llama 2.
        % Params: número de parámetros en miles de millones (B).
        % HallRate: tasa de alucinación reportada (0‑1).
        Model   = {'Llama2‑7B'; 'Llama2‑13B'; 'Llama2‑30B'; 'Llama2‑70B'};
        Params  = [7; 13; 30; 70];
        HallRate= [0.056; 0.059; 0.033; 0.051];
        T = table(Model, Params, HallRate);
        writetable(T, dataFile);
    end

    % Variable explicativa: log10 de los parámetros
    T.LogParams = log10(T.Params);

    %% 2. Preparación de variables
    X_full = T.LogParams;
    Y_full = T.HallRate;

    %% 3. Ajuste de modelos de regresión y validación LOOCV
    % Debido al pequeño tamaño del dataset, se utiliza Leave-One-Out Cross-Validation (LOOCV)
    % en lugar de una división train/test fija.

    n_samples = height(T);
    Y_pred_lin_loocv = zeros(n_samples, 1);
    Y_pred_quad_loocv = zeros(n_samples, 1);

    for i = 1:n_samples
        % Índices para esta iteración de LOOCV
        idxTrain = true(n_samples, 1);
        idxTrain(i) = false;
        idxTest = ~idxTrain;

        X_train = X_full(idxTrain);
        Y_train = Y_full(idxTrain);
        X_test = X_full(idxTest);

        % Ajuste de modelos con el subconjunto de entrenamiento
        mdl_lin_loocv = fitlm(X_train, Y_train, 'linear', 'Intercept', true);
        mdl_quad_loocv = fitlm([X_train X_train.^2], Y_train, 'Intercept', true);

        % Predicción sobre el dato de prueba
        Y_pred_lin_loocv(i) = predict(mdl_lin_loocv, X_test);
        Y_pred_quad_loocv(i) = predict(mdl_quad_loocv, [X_test X_test^2]);
    end

    % Ajuste de modelos con todos los datos para análisis de coeficientes y gráficos
    mdl_lin = fitlm(X_full, Y_full, 'linear', 'Intercept', true);
    mdl_quad = fitlm([X_full X_full.^2], Y_full, 'Intercept', true);

    % Coeficientes y intervalos de confianza (del modelo completo)
    coef_lin  = mdl_lin.Coefficients;
    ci_lin    = coefCI(mdl_lin, 0.05);
    coef_quad = mdl_quad.Coefficients;
    ci_quad   = coefCI(mdl_quad, 0.05);

    % Predicciones en el conjunto de entrenamiento completo
    Y_pred_lin_train  = predict(mdl_lin, X_full);
    Y_pred_quad_train = predict(mdl_quad, [X_full X_full.^2]);

    % Métricas de desempeño (train - sobre el modelo completo)
    metrics_lin_train  = compute_metrics(Y_full, Y_pred_lin_train, mdl_lin.NumEstimatedCoefficients);
    metrics_quad_train = compute_metrics(Y_full, Y_pred_quad_train, mdl_quad.NumEstimatedCoefficients);
    % Métricas de desempeño (test - a partir de LOOCV)
    metrics_lin_test   = compute_metrics(Y_full, Y_pred_lin_loocv, mdl_lin.NumEstimatedCoefficients);
    metrics_quad_test  = compute_metrics(Y_full, Y_pred_quad_loocv, mdl_quad.NumEstimatedCoefficients);

    % Predicciones sobre una rejilla para graficar (usando modelo completo)
    x_grid  = linspace(min(X_full)*0.95, max(X_full)*1.05, 100).';
    y_lin   = predict(mdl_lin, x_grid);
    y_quad  = predict(mdl_quad, [x_grid x_grid.^2]);

    %% 4. Exportación de tablas de coeficientes y métricas
    % Preparar tabla de coeficientes para modelo lineal
    coef_table_lin = table(coef_lin.Properties.RowNames, coef_lin.Estimate, coef_lin.SE, coef_lin.tStat, coef_lin.pValue, ...
                           ci_lin(:,1), ci_lin(:,2), 'VariableNames',{'Term','Estimate','StdError','tStat','pValue','CI_Lower','CI_Upper'});
    writetable(coef_table_lin, 'coef_lineal.csv');

    % Tabla de coeficientes para modelo cuadrático
    coef_table_quad = table(coef_quad.Properties.RowNames, coef_quad.Estimate, coef_quad.SE, coef_quad.tStat, coef_quad.pValue, ...
                            ci_quad(:,1), ci_quad(:,2), 'VariableNames',{'Term','Estimate','StdError','tStat','pValue','CI_Lower','CI_Upper'});
    writetable(coef_table_quad, 'coef_cuadratico.csv');

    % Tabla de métricas
    metrics_table = table({'Linear';'Quadratic'}, ...
                          [metrics_lin_train.R2; metrics_quad_train.R2], ...
                          [metrics_lin_train.R2adj; metrics_quad_train.R2adj], ...
                          [metrics_lin_train.RMSE; metrics_quad_train.RMSE], ...
                          [metrics_lin_train.MAE; metrics_quad_train.MAE], ...
                          [metrics_lin_train.AIC; metrics_quad_train.AIC], ...
                          [metrics_lin_train.BIC; metrics_quad_train.BIC], ...
                          [metrics_lin_test.RMSE; metrics_quad_test.RMSE], ...
                          [metrics_lin_test.MAE; metrics_quad_test.MAE], ...
                          'VariableNames',{'Model','R2','R2_adj','RMSE_train','MAE_train','AIC','BIC','RMSE_test','MAE_test'});
    writetable(metrics_table, 'metricas_modelos.csv');

    % Tabla de predicciones y residuales
    % Añadir predicciones de ambos modelos y residuales para todo el conjunto
    Y_pred_lin_all  = predict(mdl_lin, X_full);
    Y_pred_quad_all = predict(mdl_quad, [X_full X_full.^2]);
    resid_lin_all   = Y_full - Y_pred_lin_all;
    resid_quad_all  = Y_full - Y_pred_quad_all;
    tabla_pred = table(T.Model, T.Params, T.LogParams, Y_full, Y_pred_lin_all, resid_lin_all, Y_pred_quad_all, resid_quad_all, ...
                       'VariableNames',{'Model','Params','LogParams','HallRate','Pred_Lin','Resid_Lin','Pred_Quad','Resid_Quad'});
    writetable(tabla_pred, 'tabla_predicciones.csv');

    %% 5. Cálculo de predicción para 40B
    params_query = 40; % modelo de 40B parámetros
    X_query = log10(params_query);
    y_query_lin  = predict(mdl_lin, X_query);
    y_query_quad = predict(mdl_quad, [X_query X_query^2]);
    fprintf('Predicción de tasa de alucinación para 40B (lineal): %.4f\n', y_query_lin);
    fprintf('Predicción de tasa de alucinación para 40B (cuadrática): %.4f\n', y_query_quad);

    %% 6. Gráficas
    % 6.1 Ajuste lineal con banda de confianza
    figure('Name','Regresión lineal');
    hold on;
    scatter(X_full, Y_full, 50, 'filled', 'MarkerFaceColor',[0.2 0.6 0.8]);
    plot(x_grid, y_lin, 'r-', 'LineWidth',1.5);
    % Calcular bandas de confianza al 95% para el modelo lineal
    [~, y_lin_ci] = predict(mdl_lin, x_grid, 'Alpha',0.05);
    fill([x_grid; flipud(x_grid)], [y_lin_ci(:,1); flipud(y_lin_ci(:,2))], [1 0.8 0.8], 'EdgeColor','none', 'FaceAlpha',0.5);
    xlabel('log_{10}(Número de parámetros)');
    ylabel('Tasa de alucinación');
    title('Modelo de regresión lineal con banda de confianza');
    legend('Datos','Ajuste lineal','Banda 95%','Location','NorthEast');
    grid on;
    saveas(gcf, 'ajuste_lineal.png');

    % 6.2 Ajuste cuadrático con banda de confianza
    figure('Name','Regresión cuadrática');
    hold on;
    scatter(X_full, Y_full, 50, 'filled', 'MarkerFaceColor',[0.2 0.6 0.8]);
    plot(x_grid, y_quad, 'm-', 'LineWidth',1.5);
    % Banda de confianza para modelo cuadrático
    [~, y_quad_ci] = predict(mdl_quad, [x_grid x_grid.^2], 'Alpha',0.05);
    fill([x_grid; flipud(x_grid)], [y_quad_ci(:,1); flipud(y_quad_ci(:,2))], [0.9 0.8 1], 'EdgeColor','none', 'FaceAlpha',0.5);
    xlabel('log_{10}(Número de parámetros)');
    ylabel('Tasa de alucinación');
    title('Modelo de regresión cuadrática con banda de confianza');
    legend('Datos','Ajuste cuadrático','Banda 95%','Location','NorthEast');
    grid on;
    saveas(gcf, 'ajuste_cuadratico.png');

    % 6.3 Residuos vs ajustados (lineal y cuadrático)
    figure('Name','Residuos');
    subplot(2,1,1);
    plot(Y_pred_lin_all, resid_lin_all, 'bo', 'MarkerFaceColor','b');
    xlabel('Valores ajustados (lineal)'); ylabel('Residuo');
    title('Residuos vs ajustados - Modelo lineal'); grid on; yline(0, 'Color', [0.5 0.5 0.5]);
    subplot(2,1,2);
    plot(Y_pred_quad_all, resid_quad_all, 'mo', 'MarkerFaceColor','m');
    xlabel('Valores ajustados (cuadrático)'); ylabel('Residuo');
    title('Residuos vs ajustados - Modelo cuadrático'); grid on; yline(0, 'Color', [0.5 0.5 0.5]);
    saveas(gcf, 'residuos_modelos.png');

    % 6.4 Predicción vs observación
    figure('Name','Predicción vs Observación');
    scatter(Y_full, Y_pred_lin_all, 50, 'filled', 'MarkerFaceColor','r'); hold on;
    scatter(Y_full, Y_pred_quad_all, 50, 'filled', 'MarkerFaceColor','m');
    plot([min(Y_full) max(Y_full)], [min(Y_full) max(Y_full)], 'm--', 'LineWidth',1);
    xlabel('Tasa de alucinación observada');
    ylabel('Tasa de alucinación predicha');
    title('Predicción vs observación');
    legend('Predicción lineal','Predicción cuadrática','Línea 45°','Location','NorthWest');
    grid on;
    saveas(gcf, 'pred_vs_obs.png');

end

%% Función auxiliar: cálculo de métricas de regresión
function m = compute_metrics(y_true, y_pred, p)
    n = numel(y_true);
    resid = y_true - y_pred;
    SSE = sum(resid.^2);
    SST = sum((y_true - mean(y_true)).^2);
    m.R2    = 1 - SSE/SST;
    m.R2adj = 1 - (1-m.R2)*(n-1)/(n-p);
    m.RMSE  = sqrt(mean(resid.^2));
    m.MAE   = mean(abs(resid));
    m.AIC   = n * log(SSE/n) + 2*p;
    m.BIC   = n * log(SSE/n) + p * log(n);
end