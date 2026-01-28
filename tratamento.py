import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Caminho
ARQUIVO = r"C:\Users\pegum\OneDrive\Desktop\pcp\sales_train_validation.csv"

# PREPARAÇÃO DOS DADOS 

print("Carregando e processando dados para diagnóstico...")
df = pd.read_csv(ARQUIVO)

# Filtro de Negócio: Loja CA_1 e Alimentos (FOODS)
df_subset = df[(df['store_id'] == 'CA_1') & (df['cat_id'] == 'FOODS')].copy()

# Transformação da serie temporal pra o formato Long
cols_id = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
df_long = pd.melt(df_subset, id_vars=cols_id, var_name='d', value_name='vendas')

# Conversão de Datas
data_inicio = datetime(2011, 1, 29)
df_long['dia_numero'] = df_long['d'].str.extract(r'(\d+)').astype(int)
df_long['data'] = data_inicio + pd.to_timedelta(df_long['dia_numero'] - 1, unit='D')

# Enriquecimento com Atributos de Tempo
df_long['ano'] = df_long['data'].dt.year
df_long['mes'] = df_long['data'].dt.month
df_long['dia_semana'] = df_long['data'].dt.day_name()

# Agregação Visão Loja
df_dia = df_long.groupby('data')['vendas'].sum().reset_index()


# 2. GERAÇÃO DOS GRÁFICOS DE CONTEXTUALIZAÇÃO

# figura com 2 linhas e 2 colunas (Painel 2x2)
fig, axs = plt.subplots(2, 2, figsize=(16, 10))
plt.subplots_adjust(hspace=0.4, wspace=0.2)

# GRÁFICO: TENDÊNCIA LONGO PRAZO 
axs[0, 0].plot(df_dia['data'], df_dia['vendas'], color='#2c3e50', linewidth=1, alpha=0.9)
axs[0, 0].set_title('A. Histórico de Demanda Agregada (2011-2016)', fontsize=12, fontweight='bold')
axs[0, 0].set_ylabel('Vendas Diárias')
axs[0, 0].grid(True, alpha=0.3)
# Formatar eixo X
axs[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axs[0, 0].xaxis.set_major_locator(mdates.YearLocator())

# GRÁFICO B: SAZONALIDADE MENSAL (Heatmap/Barra)
venda_mes = df_long.groupby('mes')['vendas'].mean()
meses = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
bars = axs[0, 1].bar(meses, venda_mes, color='#3498db')
axs[0, 1].set_title('B. Sazonalidade Anual (Média de Vendas por Mês)', fontsize=12, fontweight='bold')
axs[0, 1].set_ylim(venda_mes.min() * 0.9, venda_mes.max() * 1.05) # Zoom para ver diferença
axs[0, 1].grid(axis='y', alpha=0.3)

# GRÁFICO C: PERFIL SEMANAL (Boxplot) ---

ordem_semana = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dados_boxplot = [df_dia[df_dia['data'].dt.day_name() == dia]['vendas'] for dia in ordem_semana]
axs[1, 0].boxplot(dados_boxplot, labels=['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sab', 'Dom'], patch_artist=True,
                  boxprops=dict(facecolor='#e67e22', color='#d35400'))
axs[1, 0].set_title('C. Ciclo Semanal (Distribuição de Vendas)', fontsize=12, fontweight='bold')
axs[1, 0].grid(True, alpha=0.3)
axs[1, 0].set_ylabel('Vendas')


# Salvar e Mostrar
plt.suptitle('DIAGNÓSTICO INICIAL DO SISTEMA PRODUTIVO - LOJA CA_1 (FOODS)', fontsize=16)
plt.savefig('diagnostico_pcp.png', dpi=300)
plt.show()

print("\nGráficos gerados!")
print("Analise a imagem 'diagnostico_pcp.png' para decidir as técnicas.")