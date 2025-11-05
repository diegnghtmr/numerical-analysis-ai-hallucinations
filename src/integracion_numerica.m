%% integracion_numerica.m
% Este script implementa la sección 6 del proyecto: integración numérica de
% datos irregulares para calcular el área bajo la curva ROC (AUC) de un
% detector de alucinaciones. Se asume que se dispone de un conjunto de
% pares (FPR, TPR) derivados de un conjunto de puntuaciones y etiquetas.
% La curva ROC se construye ordenando los puntos por FPR y TPR de forma
% monótona creciente. La integral bajo la curva se aproxima primero
% utilizando la regla del trapecio sobre una malla irregular y después
% mediante la regla de Simpson 1/3 compuesta sobre una malla uniforme
% obtenida mediante interpolación monótona (PCHIP).

% Autores: [Diego Flores y Juan Mora]
% Fecha: 20‑oct‑2025

function integracion_numerica()
    % Configuración general
    rng(42); % Semilla para reproducibilidad de datos sintéticos

    %% 1. Lectura o generación de datos ROC
    % Si existe un archivo ROC de datos reales llamado 'roc_data.csv' en la
    % carpeta compartida, se carga. De lo contrario se genera un conjunto
    % sintético que emula una curva ROC típica de un detector de
    % alucinaciones: FPR en [0,1] y TPR monótonamente creciente.
    dataFile = fullfile('roc_data.csv');
    if exist(dataFile, 'file')
        data = readmatrix(dataFile);
        fpr = data(:,1);
        tpr = data(:,2);
    else
        % Generar datos sintéticos: 50 puntos irregulares
        npts = 50;
        fpr = sort(rand(npts,1));
        % TPR basado en una función sigmoide desplazada y ruido suave
        tpr_base = 1./(1 + exp(-10*(fpr - 0.4)));
        noise = 0.02*randn(npts,1);
        tpr = tpr_base + noise;
        % Forzar límites [0,1] y monotonicidad
        tpr = max(min(tpr,1),0);
        % Hacer monotónico no decreciente
        for i=2:npts
            if tpr(i) < tpr(i-1)
                tpr(i) = tpr(i-1);
            end
        end
        % Añadir puntos extremos (0,0) y (1,1) si no existen
        if fpr(1) > 0
            fpr = [0; fpr]; tpr = [0; tpr];
        end
        if fpr(end) < 1
            fpr = [fpr; 1]; tpr = [tpr; 1];
        end
        % Guardar datos para referencia
        writematrix([fpr tpr], dataFile);
    end

    % Asegurar FPR y TPR monotónicos no decrecientes
    [fpr, idx] = sort(fpr);
    tpr = tpr(idx);
    for i=2:length(tpr)
        if tpr(i) < tpr(i-1)
            tpr(i) = tpr(i-1);
        end
    end

    %% 2. Cálculo del AUC con regla del trapecio sobre malla irregular
    auc_trap = trapz_irregular(fpr, tpr);

    %% 3. Cálculo del AUC con regla de Simpson sobre malla uniforme
    % La regla de Simpson requiere un número de subintervalos par. Se
    % evaluará el AUC para diferentes densidades de malla (M), desde 21
    % hasta 1001 puntos uniformes. Se utiliza la interpolación 'pchip' para
    % reconstruir la TPR en la malla uniforme.
    M_values = [21 51 101 201 501 1001];
    auc_simpson = zeros(size(M_values));
    for k = 1:length(M_values)
        M = M_values(k);
        auc_simpson(k) = simpson_uniform(fpr, tpr, M);
    end

    %% 4. Exportación de tabla con los resultados de AUC
    tabla_auc = table(M_values', auc_simpson', 'VariableNames',{'M','AUC_Simpson'});
    tabla_auc.Trapecio_AUC = auc_trap*ones(height(tabla_auc),1);
    writetable(tabla_auc, 'tabla_auc_M.csv');

    %% 5. Cálculo de AUC acumulada (integral parcial) sobre malla irregular
    % La función cumtrapz proporciona el valor de la integral acumulada en
    % cada punto fpr(i). Se normaliza por el valor final para obtener la
    % fracción del área acumulada.
    auc_cumulative = cumtrapz(fpr, tpr);
    auc_total = auc_cumulative(end);
    auc_normalized = auc_cumulative/auc_total;

    %% 6. Gráficas
    % 6.1 Curva ROC con sombreado y área
    figure('Name', 'Curva ROC y AUC', 'Position', [100, 100, 700, 700]);
    area(fpr, tpr, 'FaceColor', [0.00 0.45 0.74], 'FaceAlpha', 0.3, 'EdgeColor', 'none'); hold on;
    plot(fpr, tpr, '-', 'LineWidth', 2.5, 'Color', [0.00 0.45 0.74], 'DisplayName', 'Curva ROC');
    plot([0 1], [0 1], '--', 'LineWidth', 1.5, 'Color', [0.5 0.5 0.5], 'DisplayName', 'Clasificador aleatorio');
    xlabel('Tasa de Falsos Positivos (FPR)', 'FontSize', 13);
    ylabel('Tasa de Verdaderos Positivos (TPR)', 'FontSize', 13);
    title(sprintf('Curva ROC (AUC = %.4f)', auc_trap), 'FontSize', 16, 'FontWeight', 'bold');
    grid on;
    axis square;
    legend('Área bajo la curva', 'Curva ROC', 'Clasificador aleatorio', 'Location', 'SouthEast', 'FontSize', 11);
    set(gca, 'FontSize', 12, 'GridAlpha', 0.3, 'LineWidth', 0.8);
    box on;
    print('roc_curve.png', '-dpng', '-r300');

    % 6.2 AUC acumulada vs FPR
    figure('Name', 'AUC acumulada', 'Position', [100, 100, 1000, 600]);
    plot(fpr, auc_normalized, '-', 'LineWidth', 2.5, 'Color', [0.85 0.33 0.10]);
    xlabel('Tasa de Falsos Positivos (FPR)', 'FontSize', 13);
    ylabel('Fracción del área acumulada', 'FontSize', 13);
    title('AUC acumulada (regla del trapecio)', 'FontSize', 16, 'FontWeight', 'bold');
    grid on;
    set(gca, 'FontSize', 12, 'GridAlpha', 0.3, 'LineWidth', 0.8);
    box on;
    print('auc_cumulative.png', '-dpng', '-r300');

    % 6.3 Convergencia del AUC de Simpson en función de M
    figure('Name', 'Convergencia Simpson', 'Position', [100, 100, 1000, 600]);
    plot(M_values, auc_simpson, '-o', 'LineWidth', 2.5, 'MarkerSize', 8, ...
         'MarkerFaceColor', [0.47 0.67 0.19], 'Color', [0.47 0.67 0.19], 'DisplayName', 'AUC (Simpson)'); hold on;
    yline(auc_trap, '--', 'LineWidth', 2, 'Color', [0.85 0.33 0.10], 'DisplayName', 'AUC (Trapecio)');
    xlabel('Número de nodos (M)', 'FontSize', 13);
    ylabel('Valor del AUC', 'FontSize', 13);
    title('Convergencia del AUC con la regla de Simpson', 'FontSize', 16, 'FontWeight', 'bold');
    legend('Location', 'best', 'FontSize', 11);
    grid on;
    set(gca, 'FontSize', 12, 'GridAlpha', 0.3, 'LineWidth', 0.8);
    box on;
    print('auc_convergence.png', '-dpng', '-r300');

    %% 7. Impresión de resultados principales
    fprintf('AUC calculado mediante regla del trapecio (malla irregular): %.6f\n', auc_trap);
    for k=1:length(M_values)
        fprintf('M=%4d -> AUC_Simpson=%.6f\n', M_values(k), auc_simpson(k));
    end
end

%% Función que calcula la integral mediante la regla del trapecio sobre
% pares de datos (x,y) ordenados de forma ascendente en x. Para mallas
% irregulares, se utiliza la fórmula general de la regla del trapecio:
% \int_a^b f(x) dx \approx \sum_{i=2}^n (f(x_{i-1})+f(x_i))/2 * (x_i - x_{i-1}).
function A = trapz_irregular(x, y)
    % Verificar dimensiones
    assert(numel(x) == numel(y), 'x y y deben tener la misma longitud');
    % Asegurar que x esté ordenado ascendentemente
    [x, idx] = sort(x(:));
    y = y(idx);
    % Calcular las diferencias de x
    dx = diff(x);
    % Calcular el término medio de y
    ym = (y(1:end-1) + y(2:end)) / 2;
    % Sumar los trapecios
    A = sum(dx .* ym);
end

%% Función que calcula la integral mediante la regla de Simpson 1/3
% compuesta. Para aplicar la regla de Simpson se requiere que la malla
% tenga M puntos uniformemente espaciados (es decir, M-1 subintervalos
% equispaciados). La función interpola los datos TPR vs FPR con PCHIP y
% luego aplica Simpson. M debe ser impar y >= 3.
function A = simpson_uniform(fpr, tpr, M)
    % Verificar que M sea impar
    if mod(M,2) == 0
        error('M debe ser impar para la regla de Simpson');
    end
    % Definir malla uniforme en [0,1]
    x_uniform = linspace(0,1,M); % nodos equiespaciados
    % Interpolar TPR en la malla uniforme usando interpolación PCHIP
    y_uniform = pchip(fpr, tpr, x_uniform);
    % Calcular paso h
    h = (x_uniform(end) - x_uniform(1)) / (M - 1);
    % Aplicar la regla de Simpson
    % f0 + fn + 4*(f1+f3+...+f_{n-1}) + 2*(f2+f4+...+f_{n-2})
    n = M - 1; % número de subintervalos
    if mod(n,2) == 1
        error('El número de subintervalos debe ser par para Simpson');
    end
    % Índices para pesos 4 y 2
    idx4 = 2:2:n;      % posiciones impares en MATLAB 1-index (x2,x4,... -> indices 2,4,...)
    idx2 = 3:2:n-1;    % posiciones pares (x3,x5,...)
    A = (h/3) * (y_uniform(1) + y_uniform(end) + 4*sum(y_uniform(idx4)) + 2*sum(y_uniform(idx2)));
end