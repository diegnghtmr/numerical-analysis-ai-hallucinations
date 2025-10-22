% root_finding_methods.m
%
% Este script demuestra la aplicación de métodos de búsqueda de raíces
% (Bisección y Newton-Raphson) para determinar el umbral óptimo en la
% detección de alucinaciones. El umbral t* se define como la raíz de
% f(t) = Precision(t) - Recall(t), es decir, el punto donde precisión y
% recuperación se igualan. Alternativamente, se compara con el valor de
% t que maximiza la F1.
% Se construye una base de datos sintética de puntuaciones (scores) y
% etiquetas (labels) para emular un detector de alucinaciones (HHEM
% v2). A partir de estas puntuaciones se calculan las métricas por
% umbral, se interpolan con PCHIP y se aplican los métodos numéricos.

function root_finding_methods()
    % ----------------------------
    % 1. Generación del dataset
    % ----------------------------
    rng(0); % reproducibilidad
    N_pos = 100; N_neg = 100;
    % Generar puntuaciones para positivos y negativos
    scores_pos = min(max(normrnd(0.7,0.1,N_pos,1),0),1);
    scores_neg = min(max(normrnd(0.3,0.1,N_neg,1),0),1);
    labels = [ones(N_pos,1); zeros(N_neg,1)];
    scores = [scores_pos; scores_neg];
    % Guardar dataset sintetico
    tbl_dataset = table(scores, labels, 'VariableNames', {'score','label'});
    writetable(tbl_dataset, 'scores_labels.csv');

    % ----------------------------
    % 2. Cálculo de curvas P-R-F1
    % ----------------------------
    [t_vals, P_vals, R_vals, F1_vals] = compute_PRF1_curves(scores, labels);
    % Guardar curvas en CSV
    tbl_curvas = table(t_vals, P_vals, R_vals, F1_vals, ...
        'VariableNames', {'threshold','Precision','Recall','F1'});
    writetable(tbl_curvas, 'curvas_PRF1.csv');

    % ----------------------------
    % 3. Interpolación para funciones continuas
    % ----------------------------
    P_interp = @(t) pchip(t_vals, P_vals, t);
    R_interp = @(t) pchip(t_vals, R_vals, t);
    F1_interp = @(t) pchip(t_vals, F1_vals, t);
    f = @(t) P_interp(t) - R_interp(t);

    % ----------------------------
    % 4. Búsqueda de bracket para la raíz
    % ----------------------------
    % Escanear malla en [0,1] para encontrar cambio de signo
    bracket = [t_vals(1), t_vals(end)];
    % Buscar valores discretos de f(t) y detectar signo
    ft_values = f(t_vals);
    sign_change_idx = find(ft_values(1:end-1).*ft_values(2:end) <= 0, 1);
    if ~isempty(sign_change_idx)
        a0 = t_vals(sign_change_idx);
        b0 = t_vals(sign_change_idx+1);
    else
        % Si no hay cambio, usar extremos del rango
        a0 = t_vals(1);
        b0 = t_vals(end);
    end

    % ----------------------------
    % 5. Aplicación de Bisección
    % ----------------------------
    tol = 1e-6; maxit = 100;
    [root_bis, hist_bis] = biseccion(f, a0, b0, tol, maxit);
    % Guardar tabla de iteraciones de bisección
    tbl_bis = array2table(hist_bis, 'VariableNames', ...
        {'iter','a','b','mid','f_mid','interval'});
    writetable(tbl_bis, 'tabla_biseccion.csv');

    % ----------------------------
    % 6. Aplicación de Newton-Raphson
    % ----------------------------
    % Punto inicial: umbral que maximiza la F1
    [~, idx_maxF1] = max(F1_vals);
    x0 = t_vals(idx_maxF1);
    h = 1e-4;
    [root_new, hist_new] = newton_method(f, x0, h, tol, maxit);
    tbl_new = array2table(hist_new, 'VariableNames', ...
        {'iter','x_k','f_x','delta'});
    writetable(tbl_new, 'tabla_newton.csv');

    % ----------------------------
    % 7. Validación y comparación
    % ----------------------------
    % Buscar umbral que maximiza F1 en malla fina
    t_fine = linspace(t_vals(1), t_vals(end), 1001);
    [~, idx_fine] = max(F1_interp(t_fine));
    t_maxF1 = t_fine(idx_fine);

    % ----------------------------
    % 8. Gráficas
    % ----------------------------
    % f(t) y raíces
    figure;
    plot(t_fine, f(t_fine), 'b-', 'LineWidth',1.2);
    hold on;
    yline(0,'k--');
    plot(root_bis, 0, 'ro', 'DisplayName','Biseccion');
    plot(root_new, 0, 'gx', 'DisplayName','Newton');
    title('Función f(t)=P(t)-R(t) y raíces estimadas');
    xlabel('Umbral t'); ylabel('f(t)');
    legend('f(t)','y=0','Bisección','Newton','Location','best'); grid on; hold off;
    saveas(gcf, 'fig_f_t_crossing.png'); close;

    % Curvas P, R, F1
    figure;
    plot(t_vals, P_vals, 'b-', 'LineWidth',1.2); hold on;
    plot(t_vals, R_vals, 'r-', 'LineWidth',1.2);
    plot(t_vals, F1_vals, 'g-', 'LineWidth',1.2);
    xline(root_bis, 'k--', 'DisplayName','t* (P=R)');
    xline(t_maxF1, 'm--', 'DisplayName','t (max F1)');
    title('Curvas de Precision, Recall y F1 vs Umbral');
    xlabel('Umbral t'); ylabel('Valor de la métrica');
    legend({'Precision','Recall','F1','t* (P=R)','t (max F1)'}, 'Location','best');
    grid on; hold off;
    saveas(gcf, 'fig_curvas_PRF1.png'); close;

    % Convergencia Bisección
    figure;
    semilogy(hist_bis(:,1), hist_bis(:,6), '-o', 'LineWidth',1.2, 'DisplayName','Bisección');
    hold on;
    semilogy(hist_new(:,1), hist_new(:,4), '-s', 'LineWidth',1.2, 'DisplayName','Newton');
    title('Convergencia de métodos de búsqueda de raíces');
    xlabel('Iteración k'); ylabel('Error aproximado');
    legend('Bisección |b-a|/2','Newton |x_k - x_{k-1}|'); grid on; hold off;
    saveas(gcf, 'fig_convergencia_metodos.png'); close;

    fprintf('Umbral raíz (Bisección): %.6f\n', root_bis);
    fprintf('Umbral raíz (Newton): %.6f\n', root_new);
    fprintf('Umbral máxima F1: %.6f\n', t_maxF1);
end

function [t_vals, P_vals, R_vals, F1_vals] = compute_PRF1_curves(scores, labels)
    % Ordenar los scores y obtener umbrales únicos
    [sorted_scores, idx] = sort(scores, 'ascend');
    sorted_labels = labels(idx);
    thresholds = unique(sorted_scores);

    n = length(scores);
    P_vals = zeros(length(thresholds),1);
    R_vals = zeros(length(thresholds),1);
    F1_vals = zeros(length(thresholds),1);

    for i = 1:length(thresholds)
        t = thresholds(i);
        % Predicciones: 1 si score >= t, 0 en caso contrario
        y_pred = scores >= t;
        tp = sum((y_pred == 1) & (labels == 1));
        fp = sum((y_pred == 1) & (labels == 0));
        fn = sum((y_pred == 0) & (labels == 1));
        % Calcular métricas evitando división por cero
        P = tp / max(tp + fp, 1e-12);
        R = tp / max(tp + fn, 1e-12);
        if P + R > 0
            F1 = 2 * P * R / (P + R);
        else
            F1 = 0;
        end
        P_vals(i) = P;
        R_vals(i) = R;
        F1_vals(i) = F1;
    end
    t_vals = thresholds;
end

function [root, hist] = biseccion(f, a, b, tol, maxit)
    % Método de Bisección para f(x)=0. Devuelve la raíz aproximada y
    % almacena el historial de iteraciones.
    fa = f(a);
    fb = f(b);
    if fa * fb > 0
        error('La función no cambia de signo en [a,b]');
    end
    hist = zeros(maxit,6);
    for k = 1:maxit
        c = (a + b) / 2;
        fc = f(c);
        hist(k,:) = [k, a, b, c, fc, (b - a)/2];
        if abs(fc) < tol || (b - a)/2 < tol
            root = c;
            hist = hist(1:k, :);
            return;
        end
        if fa * fc < 0
            b = c; fb = fc;
        else
            a = c; fa = fc;
        end
    end
    root = c;
    hist = hist(1:maxit, :);
end

function [root, hist] = newton_method(f, x0, h, tol, maxit)
    % Método de Newton-Raphson con derivada numérica central. Devuelve
    % la raíz aproximada y el historial de iteraciones.
    x_prev = x0;
    hist = zeros(maxit,4);
    for k = 1:maxit
        fx = f(x_prev);
        % Derivada numérica central
        dfx = (f(x_prev + h) - f(x_prev - h)) / (2*h);
        % Evitar divisiones por valores muy pequeños
        if abs(dfx) < 1e-8
            % Salir anticipadamente si la derivada es muy pequeña
            root = x_prev;
            hist = hist(1:k-1,:);
            return;
        end
        x_new = x_prev - fx / dfx;
        delta = abs(x_new - x_prev);
        hist(k,:) = [k, x_new, fx, delta];
        if abs(f(x_new)) < tol || delta < tol
            root = x_new;
            hist = hist(1:k,:);
            return;
        end
        x_prev = x_new;
    end
    root = x_prev;
    hist = hist(1:maxit,:);
end
