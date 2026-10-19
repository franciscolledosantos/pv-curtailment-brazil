"""Descrição dos cortes de fotovoltaicas (ONS, abr/2024–out/2026) e relação com carga líquida do sistema."""
import os, numpy as np, pandas as pd
import statsmodels.formula.api as smf
B = os.path.join(os.path.dirname(__file__), '..', 'dados'); OUT = os.path.join(os.path.dirname(__file__), '..', 'resultados')
H = pd.read_parquet(os.path.join(B, 'painel_pv_horario.parquet')); S = pd.read_parquet(os.path.join(B, 'sistema_horario.parquet'))
H['mes'] = H['hora'].dt.to_period('M'); H['h'] = H['hora'].dt.hour
C = ['ger', 'gnr', 'pot', 'gnr_REL', 'gnr_CNF', 'gnr_ENE', 'gnr_LOC']
M = H.groupby('mes')[C].sum() / 1e3
for c in ['gnr', 'gnr_REL', 'gnr_CNF', 'gnr_ENE']:
    M['pct_' + c] = 100 * M[c] / M['pot']
M['n_usinas'] = H[H['pot'] > 0].groupby('mes')['id_ons'].nunique()
M.to_csv(os.path.join(OUT, 'mensal.csv'))
Y = H.assign(ano=H['hora'].dt.year).groupby('ano')[C].sum() / 1e3
Y['pct'] = 100 * Y['gnr'] / Y['pot']; Y.to_csv(os.path.join(OUT, 'anual.csv')); print(Y.round(1))
Hh = H.groupby('h')[C].sum(); Hh = 100 * Hh[['gnr', 'gnr_REL', 'gnr_CNF', 'gnr_ENE']].div(Hh['pot'], axis=0); Hh.to_csv(os.path.join(OUT, 'perfil_horario_pct.csv'))
U = H.groupby('uf')[C].sum() / 1e3; U['pct'] = 100 * U['gnr'] / U['pot']; U['pct_grid'] = 100 * (U['gnr_REL'] + U['gnr_CNF']) / U['pot']
U.sort_values('gnr', ascending=False).to_csv(os.path.join(OUT, 'por_uf.csv')); print(U.sort_values('gnr', ascending=False).round(1))
P = H.groupby('id_ons')[C].sum(); P['grid'] = P['gnr_REL'] + P['gnr_CNF']
P['pct'] = 100 * P['gnr'] / P['pot']; P['pct_grid'] = 100 * P['grid'] / P['pot']
P.to_csv(os.path.join(OUT, 'por_usina.csv'))
# concentração do corte por rede: Gini e parcela das 10 usinas com maior corte por rede
g = np.sort(P['grid'].values); n = len(g); gini = (2 * np.sum((np.arange(1, n + 1)) * g) / (n * g.sum())) - (n + 1) / n
top10 = P['grid'].nlargest(10).sum() / P['grid'].sum()
ge = np.sort(P['gnr_ENE'].values); gini_e = (2 * np.sum((np.arange(1, n + 1)) * ge) / (n * ge.sum())) - (n + 1) / n
print('Gini corte de rede', round(gini, 2), 'top10', round(top10, 2), 'Gini ENE', round(gini_e, 2))
pc = H.groupby('ponto')[['gnr_REL', 'gnr_CNF', 'pot']].sum(); pc['grid'] = pc['gnr_REL'] + pc['gnr_CNF']
pc['pct_grid'] = 100 * pc['grid'] / pc['pot']; pc.sort_values('grid', ascending=False).to_csv(os.path.join(OUT, 'por_ponto_conexao.csv'))
# relação diária: fração ENE do sistema vs carga líquida (meio-dia, 10-15h)
d = H[H['h'].between(10, 15)].groupby(H['hora'].dt.normalize())[['gnr_ENE', 'gnr_REL', 'gnr_CNF', 'pot']].sum()
Sm = S[S.index.hour.isin(range(10, 16))]; s = Sm.groupby(Sm.index.normalize())[['carga_SIN', 'eol_SIN', 'pv_SIN', 'liq_SIN']].mean()
d = d.join(s, how='inner'); d['f_ene'] = d['gnr_ENE'] / d['pot']; d['f_grid'] = (d['gnr_REL'] + d['gnr_CNF']) / d['pot']
d['fds'] = (d.index.dayofweek >= 5).astype(int); d['t'] = (d.index - d.index.min()).days / 365.25
d['liq_GW'] = d['liq_SIN'] / 1e3; d['eol_GW'] = d['eol_SIN'] / 1e3; d['carga_GW'] = d['carga_SIN'] / 1e3; d['pv_GW'] = d['pv_SIN'] / 1e3
d.to_csv(os.path.join(OUT, 'diario_sistema.csv'))
out = {}
for y in ['f_ene', 'f_grid']:
    m = smf.ols(f'{y} ~ carga_GW + eol_GW + pv_GW + fds + t', d).fit(cov_type='HAC', cov_kwds={'maxlags': 7})
    print(y, m.params.round(4).to_dict(), 'R2', round(m.rsquared, 2)); out[y] = m
    pd.DataFrame({'coef': m.params, 'se': m.bse, 'p': m.pvalues}).to_csv(os.path.join(OUT, f'reg_diaria_{y}.csv'))
    out[y + '_r2'] = m.rsquared
print('corr f_ene x liq', d[['f_ene', 'liq_GW']].corr().iloc[0, 1].round(2), 'corr f_grid x liq', d[['f_grid', 'liq_GW']].corr().iloc[0, 1].round(2))
