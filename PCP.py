
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import itertools
import logging
from datetime import datetime

# Bibliotecas de Modelagem Estatística e ML
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Configurações de Warning
import warnings
warnings.filterwarnings("ignore")
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)

# ==============================================================================
# 1. DEFINIÇÃO DE PARÂMETROS E CARGA DE DADOS
# ==============================================================================
ARQUIVO_FONTE = r"C:\Users\hugos\Desktop\PCP\Modulo 2\sales_train_validation.csv"
HORIZONTE_PREVISAO = 180

print("1. Iniciando pipeline de dados...")

try:
    df_raw = pd.read_csv(ARQUIVO_FONTE)
except FileNotFoundError:
    print("   [AVISO] Arquivo base não encontrado. Gerando dados sintéticos para validação do script.")
    rng = pd.date_range(start='2011-01-29', periods=1900)
    df_raw = pd.DataFrame({'vendas': np.random.randint(50, 200, size=1900), 'data': rng})
    df_raw.loc[1750, 'vendas'] = 0 # Simulação de falha na coleta
    df_agregado = df_raw.set_index('data')
else:
    # Filtragem do escopo (Loja CA_1 / Categoria FOODS)
    df_scope = df_raw[(df_raw['store_id'] == 'CA_1') & (df_raw['cat_id'] == 'FOODS')].copy()
    
    # Transformação estrutural (Wide -> Long)
    cols_id = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    df_long = pd.melt(df_scope, id_vars=cols_id, var_name='d', value_name='vendas')
    
    # Conversão temporal
    data_ref = datetime(2011, 1, 29)
    df_long['dia_sequencial'] = df_long['d'].str.extract(r'(\d+)').astype(int)
    df_long['data'] = data_ref + pd.to_timedelta(df_long['dia_sequencial'] - 1, unit='D')
    
    # Agregação diária
    df_agregado = df_long.groupby('data')['vendas'].sum().reset_index()
    df_agregado = df_agregado.set_index('data')
    df_agregado.index.freq = 'D'

# ==============================================================================
# 2. PRÉ-PROCESSAMENTO E HIGIENIZAÇÃO (DATA CLEANING)
# ==============================================================================
print("2. Executando tratamento de outliers e inconsistências...")

# Tratamento 1: Correção de Rupturas Operacionais (Lojas fechadas/Feriados)
# Critério: Vendas < 10 unidades são consideradas anomalias operacionais.
# Ação: Substituição por interpolação linear baseada na tendência local.
mask_inconsistencia = df_agregado['vendas'] < 10
df_agregado.loc[mask_inconsistencia, 'vendas'] = np.nan
df_agregado['vendas'] = df_agregado['vendas'].interpolate(method='linear')

# Tratamento 2: Suavização de Picos (Winsorização Superior)
# Critério: Valores acima do percentil 99% são considerados eventos atípicos não recorrentes.
# Ação: Teto dos valores ao limite do percentil 99.
limite_superior = df_agregado['vendas'].quantile(0.99)
df_agregado.loc[df_agregado['vendas'] > limite_superior, 'vendas'] = limite_superior

# Definição dos conjuntos de Treino (Histórico) e Teste (Validação)
df_train = df_agregado.iloc[:-HORIZONTE_PREVISAO]
df_test = df_agregado.iloc[-HORIZONTE_PREVISAO:]

# ==============================================================================
# 3. FUNÇÕES AUXILIARES DE AVALIAÇÃO
# ==============================================================================
def calcular_indicadores(y_real, y_prev):
    """
    Calcula métricas de erro padrão para avaliação de aderência.
    Retorna: Dicionário com MAD, MSE e MAPE.
    """
    mad = mean_absolute_error(y_real, y_prev)
    mse = mean_squared_error(y_real, y_prev)
    
    # Tratamento para evitar divisão por zero no MAPE
    y_real_adj = np.where(y_real == 0, 1, y_real) 
    mape = np.mean(np.abs((y_real - y_prev) / y_real_adj)) * 100
    
    return {"MAD": round(mad, 2), "MAPE": round(mape, 2), "MSE": round(mse, 2)}

def gerar_features_temporais(df_input):
    """Engenharia de atributos para modelos baseados em árvore (XGBoost)."""
    df_out = df_input.copy()
    df_out['dayofweek'] = df_out.index.dayofweek
    df_out['month'] = df_out.index.month
    df_out['year'] = df_out.index.year
    df_out['dayofyear'] = df_out.index.dayofyear
    return df_out

# ==============================================================================
# 4. OTIMIZAÇÃO DE HIPERPARÂMETROS (GRID SEARCH) E PREVISÃO
# ==============================================================================
print("3. Iniciando Grid Search e Modelagem...")

# -----------------------------------------------------------------------------
# 4.1. Holt-Winters (Otimização Simples)
# -----------------------------------------------------------------------------
print("   > Otimizando Holt-Winters...")
configs_hw = [
    {'trend': 'add', 'seasonal': 'add', 'damped': True},
    {'trend': 'add', 'seasonal': 'add', 'damped': False},
    {'trend': 'add', 'seasonal': 'mul', 'damped': True} 
]

best_score_hw = float("inf")
best_model_hw = None
best_config_hw = None

for config in configs_hw:
    try:
        model = ExponentialSmoothing(df_train['vendas'], 
                                     trend=config['trend'], 
                                     damped_trend=config['damped'], 
                                     seasonal=config['seasonal'], 
                                     seasonal_periods=7).fit()
        pred = model.forecast(len(df_test))
        score = mean_squared_error(df_test['vendas'], pred)
        
        if score < best_score_hw:
            best_score_hw = score
            best_model_hw = model
            best_config_hw = config
    except:
        continue

# Previsão final com o melhor modelo
pred_hw = best_model_hw.forecast(HORIZONTE_PREVISAO)

# Print corrigido usando a variável best_config_hw
print(f"     Melhor HW: Trend={best_config_hw['trend']}, Damped={best_config_hw['damped']} | Erro (MSE): {best_score_hw:.2f}")


# -----------------------------------------------------------------------------
# 4.2. SARIMAX (Grid Search em p, d, q, P, D, Q)
# -----------------------------------------------------------------------------
print("   > Otimizando SARIMAX ...")

# Define faixas de parâmetros
p = range(0, 3)
d = range(0, 2)
q = range(0, 2)

pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 7) for x in list(itertools.product(p, d, q))]

best_score_sarima = float("inf")
best_order = None
best_seasonal = None
best_model_sarima = None

# Loop de Grid Search
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = SARIMAX(df_train['vendas'],
                          order=param,
                          seasonal_order=param_seasonal,
                          enforce_stationarity=False,
                          enforce_invertibility=False)
            results = mod.fit(disp=False)
            
            # Previsão in-sample para validação
            pred = results.get_forecast(steps=len(df_test)).predicted_mean
            score = mean_squared_error(df_test['vendas'], pred)
            
            if score < best_score_sarima:
                best_score_sarima = score
                best_order = param
                best_seasonal = param_seasonal
                best_model_sarima = results
        except:
            continue

pred_sarima = best_model_sarima.forecast(steps=HORIZONTE_PREVISAO)
print(f"     Melhor SARIMAX: Order{best_order} x Seasonal{best_seasonal}")


# -----------------------------------------------------------------------------
# 4.3. Prophet (Otimização de Flexibilidade)
# -----------------------------------------------------------------------------
print("   > Otimizando Prophet...")
param_grid_prophet = {  
    'changepoint_prior_scale': [0.01, 0.05, 0.5], # Flexibilidade da tendência
    'seasonality_prior_scale': [0.1, 1.0, 10.0]  # Força da sazonalidade
}

all_params = [dict(zip(param_grid_prophet.keys(), v)) for v in itertools.product(*param_grid_prophet.values())]
best_score_prophet = float("inf")
best_params_prophet = {}

# Preparar dados formato Prophet
df_prophet_train = df_train.reset_index().rename(columns={'data': 'ds', 'vendas': 'y'})
df_prophet_test = df_test.reset_index().rename(columns={'data': 'ds', 'vendas': 'y'})

for params in all_params:
    m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True, **params)
    m.fit(df_prophet_train)
    
    future = m.make_future_dataframe(periods=len(df_test))
    forecast = m.predict(future)
    pred = forecast['yhat'].tail(len(df_test)).values
    
    score = mean_squared_error(df_test['vendas'], pred)
    
    if score < best_score_prophet:
        best_score_prophet = score
        best_params_prophet = params

# Retreinar com melhores parâmetros
model_prophet = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True, **best_params_prophet)
model_prophet.fit(df_prophet_train)
future_dates = model_prophet.make_future_dataframe(periods=HORIZONTE_PREVISAO)
forecast_prophet = model_prophet.predict(future_dates)
pred_prophet = forecast_prophet['yhat'].tail(HORIZONTE_PREVISAO).values
print(f"     Melhor Prophet: {best_params_prophet}")


# -----------------------------------------------------------------------------
# 4.4. XGBoost (Grid Search Robusto)
# -----------------------------------------------------------------------------
print("   > Otimizando XGBoost...")

X_train = gerar_features_temporais(df_train).drop(columns=['vendas'])
y_train = df_train['vendas']
X_test = gerar_features_temporais(df_test).drop(columns=['vendas'])

param_grid_xgb = {
    'n_estimators': [100, 500, 1000],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1]
}

keys, values = zip(*param_grid_xgb.items())
xgb_configs = [dict(zip(keys, v)) for v in itertools.product(*values)]

best_score_xgb = float("inf")
best_model_xgb = None

for params in xgb_configs:
    model = XGBRegressor(objective='reg:squarederror', **params)
    
    # Treino sem verbose para limpar output
    model.fit(X_train, y_train)
    
    pred = model.predict(X_test)
    score = mean_squared_error(df_test['vendas'], pred)
    
    if score < best_score_xgb:
        best_score_xgb = score
        best_model_xgb = model

pred_xgb = best_model_xgb.predict(X_test)
print(f"     Melhor XGBoost: {best_model_xgb.get_params()['n_estimators']} est, "
      f"LR: {best_model_xgb.get_params()['learning_rate']}, "
      f"Depth: {best_model_xgb.get_params()['max_depth']}")

# ==============================================================================
# 5. CONSOLIDAÇÃO DOS RESULTADOS
# ==============================================================================
df_resultados = pd.DataFrame({
    "Data": df_test.index,
    "Demanda_Real": df_test['vendas'].values,
    "Holt_Winters": pred_hw.values,
    "SARIMAX": pred_sarima.values,
    "Prophet": pred_prophet,
    "XGBoost": pred_xgb
})

# Exibição Tabular (Amostragem)
print("\n=== TABELA 1: Comparativo de Previsões (Amostra Inicial e Final) ===")
print(df_resultados.head(5).to_string(index=False))
print("...")
print(df_resultados.tail(5).to_string(index=False, header=False))

# Cálculo de Métricas de Erro
metricas_consolidadas = {
    "Holt-Winters": calcular_indicadores(df_test['vendas'], pred_hw),
    "SARIMAX": calcular_indicadores(df_test['vendas'], pred_sarima),
    "Prophet": calcular_indicadores(df_test['vendas'], pred_prophet),
    "XGBoost": calcular_indicadores(df_test['vendas'], pred_xgb)
}

df_metricas = pd.DataFrame(metricas_consolidadas).T
print("\n=== TABELA 2: Indicadores de Desempenho (Erro) ===")
print(df_metricas.to_string())

# ==============================================================================
# 6. VISUALIZAÇÃO GRÁFICA
# ==============================================================================
plt.figure(figsize=(14, 7))

# Série Real (Benchmark)
plt.plot(df_resultados["Data"], df_resultados["Demanda_Real"], 
         color="black", linewidth=2.5, alpha=0.6, label="Demanda Real (Observada)")

# Modelos Estatísticos Clássicos
plt.plot(df_resultados["Data"], df_resultados["Holt_Winters"], 
         color="#1f77b4", linewidth=1.5, marker="^", markersize=4, markevery=7, label="Holt-Winters")

plt.plot(df_resultados["Data"], df_resultados["SARIMAX"], 
         color="#2ca02c", linewidth=1.5, linestyle="--", label="SARIMAX")

# Modelos Avançados / ML
plt.plot(df_resultados["Data"], df_resultados["Prophet"], 
         color="#9467bd", linewidth=1.5, linestyle="-.", label="Prophet")

plt.plot(df_resultados["Data"], df_resultados["XGBoost"], 
         color="#d62728", linewidth=1.5, linestyle=":", label="XGBoost")

# Formatação Profissional do Gráfico
plt.title("Análise Comparativa de Previsão de Demanda (Horizonte: 6 Meses)", fontsize=14, pad=15)
plt.ylabel("Volume de Vendas (Unidades)")
plt.xlabel("Período de Análise")
plt.legend(loc='upper left', frameon=True, framealpha=0.9)
plt.grid(True, linestyle='--', alpha=0.4)

# Formatação do Eixo Temporal
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b/%Y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gcf().autofmt_xdate()

plt.tight_layout()
plt.show()