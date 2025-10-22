# Análisis Numérico de Alucinaciones en Modelos de Inteligencia Artificial

Proyecto para estudiar, cuantificar y mitigar alucinaciones en modelos de IA mediante técnicas de análisis numérico (interpolación, integración, aproximaciones de Taylor, búsqueda de raíces y solución de EDOs), con generación de datos y visualizaciones reproducibles.

## Estructura del proyecto

```
Analisis-Numerico-Alucinaciones-IA/
├── src/          # Scripts .m (MATLAB/Octave)
├── data/         # Archivos .csv generados
├── results/      # Gráficas .png generadas
├── docs/         # Documentación adicional
├── README.md     # Este archivo
├── LICENSE       # Licencia MIT
└── .gitignore    # Archivos ignorados por Git
```

## Requisitos

- **MATLAB** R2021b o superior (recomendado) o **GNU Octave** 7+
- **Git** 2.0+
- (Opcional) **Visual Studio Code** con extensión MATLAB

## Instalación

1. Clonar este repositorio:
   ```bash
   git clone https://github.com/TU_USUARIO/Analisis-Numerico-Alucinaciones-IA.git
   cd Analisis-Numerico-Alucinaciones-IA
   ```

2. Abrir en MATLAB o VSCode con extensión MATLAB instalada.

## Uso

### Ejecutar scripts desde MATLAB

```matlab
run('src/analisis_regresion.m')
run('src/taylor_approximation.m')
run('src/integracion_numerica.m')
run('src/interpolacion_polinomica.m')
run('src/root_finding_methods.m')
run('src/solucion_edos.m')
run('src/comparacion_precision_costos.m')
```

### Ejecutar desde terminal (con MATLAB en PATH)

```bash
matlab -batch "run('src/analisis_regresion.m')"
```

### Ejecutar con GNU Octave

```bash
octave-cli --silent --eval "run('src/analisis_regresion.m')"
```

## Descripción de scripts

| Script | Descripción |
|--------|-------------|
| **taylor_approximation.m** | Aproximaciones de series de Taylor para funciones de activación (sigmoid/tanh) y análisis de error |
| **root_finding_methods.m** | Métodos de bisección y Newton-Raphson para encontrar umbrales óptimos de detección |
| **interpolacion_polinomica.m** | Interpolación polinómica (Lagrange y Newton) para tasas de alucinación |
| **integracion_numerica.m** | Cálculo de AUC (área bajo la curva ROC) usando reglas del trapecio y Simpson |
| **solucion_edos.m** | Solución de EDOs con métodos de Euler y Runge-Kutta para modelar confianza temporal |
| **analisis_regresion.m** | Modelos de regresión lineal y cuadrática para predecir tasas de alucinación |
| **comparacion_precision_costos.m** | Comparación de precisión vs. costo computacional de todos los métodos |

## Salidas generadas

- **data/**: Archivos CSV con resultados numéricos, coeficientes, métricas y tablas
- **results/**: Gráficas PNG con visualizaciones de curvas, errores y convergencia

## Notas técnicas

- El nombre de la carpeta usa guiones (sin acentos ni espacios) para máxima compatibilidad con Git y líneas de comando
- Los archivos generados (`.csv`, `.png`) están en `.gitignore` para evitar saturar el repositorio
- Los scripts están documentados en español con comentarios detallados
- Se recomienda ejecutar los scripts en el orden listado para una mejor comprensión del flujo analítico

## Autores

- **Diego Alejandro Flores Quintero** — Universidad del Quindío
- Fecha: Octubre 2025

## Licencia

Este proyecto está licenciado bajo la [Licencia MIT](LICENSE).
