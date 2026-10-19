"""Previsão para o dia seguinte (D+1) da fração cortada por usina e hora.

Validação em janela expansível mensal: treina com tudo antes do mês m e testa no mês m (abr/2025–set/2026).
Alvos: fração total cortada, fração por razão de rede (REL+CNF) e por razão energética (ENE).
Modelos:
  P  persistência: mesma hora de D-1
  C  climatologia: média de treino por usina × hora × mês do ano (recai em usina × hora)
  L  LightGBM com informação disponível ao fim de D-1 (defasagens da usina e do sistema, calendário)
  LO LightGBM + carga, eólica e solar do sistema em D (previsão perfeita; limite superior do valor de previsões)
Métricas: erro absoluto em energia (MWh) e habilidade sobre a persistência; para o evento "corte ≥ 10% do potencial":
AUC e Brier skill score sobre a climatologia.
"""
import os, warnings
import numpy as np, pandas as pd, lightgbm as lgb, holidays
from sklearn.metrics import roc_auc_score, brier_score_loss
warnings.filterwarnings('ignore')
B = os.path.join(os.path.dirname(__file__), '..', 'dados')
OUT = os.path.join(os.path.dirname(__file__), '..', 'resultados')
H = pd.read_parquet(os.path.join(B, 'painel_pv_horario.parquet'))
S = pd.read_parquet(os.path.join(B, 'sistema_horario.parquet'))
H['h'] = H['hora'].dt.hour
H = H[H['h'].between(6, 18)].copy()
H['dia'] = H['hora'].dt.normalize()
H['f'] = np.where(H['pot'] > 0.5, (H['gnr'] / H['pot']).clip(0, 1), np.nan)
H['f_grid'] = np.where(H['pot'] > 0.5, ((H['gnr_REL'] + H['gnr_CNF']) / H['pot']).clip(0, 1), np.nan)
H['f_ene'] = np.where(H['pot'] > 0.5, (H['gnr_ENE'] / H['pot']).clip(0, 1), np.nan)
H = H.sort_values(['id_ons', 'hora'])
# defasagens por usina (calendário completo para que shift(24h) seja a mesma hora de D-1)
H = H.set_index(['id_ons', 'hora'])
idx = pd.MultiIndex.from_product([H.index.levels[0], pd.date_range(H.index.get_level_values(1).min(), H.index.get_level_values(1).max(), freq='h')], names=['id_ons', 'hora'])
G = H.reindex(idx)
for c in ['f', 'f_grid', 'f_ene', 'pot']:
    g = G[c].groupby(level=0)
    G[c + '_l24'] = g.shift(24)
    G[c + '_l48'] = g.shift(48)
    G[c + '_l168'] = g.shift(168)
    # média dos últimos 7 dias na mesma hora (D-1 ... D-7)
    G[c + '_m7'] = sum(g.shift(24 * k) for k in range(1, 8)) / 7
# média diária de D-1 (horas diurnas)
G['dia'] = G.index.get_level_values(1).normalize()
dmean = G.groupby([G.index.get_level_values(0), 'dia'])[['f', 'f_grid', 'f_ene']].mean()
dmean = dmean.groupby(level=0).shift(1).add_suffix('_d1')
G = G.join(dmean, on=[G.index.get_level_values(0), 'dia']) if False else G.reset_index().merge(
    dmean.reset_index().rename(columns={'level_0': 'id_ons', 'id_ons': 'id_ons'}), on=['id_ons', 'dia'], how='left')
G = G[G['f'].notna()].copy()
# sistema: defasagens de D-1 e valores de D (previsão perfeita)
S = S.copy()
S['ene_frac_SIN'] = np.nan
pvsys = H.reset_index().groupby('hora')[['gnr', 'gnr_ENE', 'pot']].sum()
S = S.join(pvsys.rename(columns={'gnr': 'pv_gnr', 'gnr_ENE': 'pv_gnrene', 'pot': 'pv_pot'}))
S['ene_frac_SIN'] = S['pv_gnrene'] / S['pv_pot']
for c in ['carga_SIN', 'carga_NE', 'carga_SE', 'eol_SIN', 'eol_NE', 'pv_SIN', 'liq_SIN', 'ene_frac_SIN']:
    S[c + '_l24'] = S[c].shift(24)
G = G.merge(S[[c for c in S.columns if c.endswith('_l24')] + ['carga_SIN', 'carga_NE', 'eol_SIN', 'eol_NE', 'pv_SIN', 'liq_SIN']],
            left_on='hora', right_index=True, how='left')
br = holidays.Brazil(years=range(2024, 2027))
G['dow'] = G['hora'].dt.dayofweek; G['mes'] = G['hora'].dt.month
G['feriado'] = G['hora'].dt.date.map(lambda d: d in br).astype(int)
G['doy_s'] = np.sin(2 * np.pi * G['hora'].dt.dayofyear / 365.25); G['doy_c'] = np.cos(2 * np.pi * G['hora'].dt.dayofyear / 365.25)
G['ym'] = G['hora'].dt.to_period('M')
for c in ['id_ons', 'uf', 'sub', 'ponto']:
    G[c + '_cat'] = G[c].astype('category')
G['cap'] = G['cap'].astype(float)
G.to_parquet(os.path.join(B, 'painel_modelo.parquet'), index=False)
print('linhas', len(G), 'usinas', G['id_ons'].nunique())

BASE = ['h', 'dow', 'mes', 'feriado', 'doy_s', 'doy_c', 'cap', 'id_ons_cat', 'uf_cat', 'sub_cat', 'ponto_cat', 'pot_l24', 'pot_m7']
LAGS = lambda y: [y + '_l24', y + '_l48', y + '_l168', y + '_m7', y + '_d1'] + [z for z in ['f_l24', 'f_grid_l24', 'f_ene_l24', 'f_m7', 'f_grid_m7', 'f_ene_m7'] if not z.startswith(y + '_')]
SYS1 = [c for c in G.columns if c.endswith('_l24') and c.split('_l24')[0] in ['carga_SIN', 'carga_NE', 'carga_SE', 'eol_SIN', 'eol_NE', 'pv_SIN', 'liq_SIN', 'ene_frac_SIN']]
SYS0 = ['carga_SIN', 'carga_NE', 'eol_SIN', 'eol_NE', 'pv_SIN', 'liq_SIN']
meses = pd.period_range('2025-04', '2026-09', freq='M')
res, imp = [], []
for y in ['f', 'f_grid', 'f_ene']:
    feats = {'L': BASE + LAGS(y) + SYS1, 'LO': BASE + LAGS(y) + SYS1 + SYS0}
    for m in meses:
        tr, te = G[(G['ym'] < m) & G[y].notna()], G[(G['ym'] == m) & G[y].notna()]
        te = te[te[y + '_l24'].notna()]
        if len(te) == 0:
            continue
        pred = {'P': te[y + '_l24'].values}
        clim = tr.groupby(['id_ons', 'h', 'mes'])[y].mean()
        clim2 = tr.groupby(['id_ons', 'h'])[y].mean()
        c = pd.Series(list(zip(te['id_ons'], te['h'], te['mes']))).map(clim)
        c = c.fillna(pd.Series(list(zip(te['id_ons'], te['h']))).map(clim2)).fillna(tr[y].mean())
        pred['C'] = c.values
        for nm, fs in feats.items():
            mdl = lgb.LGBMRegressor(objective='cross_entropy', n_estimators=400, learning_rate=0.05, num_leaves=63,
                                    min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
            mdl.fit(tr[fs], tr[y], sample_weight=tr['pot'].clip(lower=0.5))
            pred[nm] = np.clip(mdl.predict(te[fs]), 0, 1)
            if m == meses[-1]:
                imp.append(pd.DataFrame({'alvo': y, 'modelo': nm, 'var': fs, 'ganho': mdl.booster_.feature_importance('gain')}))
        w = te['pot'].values; obs = te[y].values
        ev = (obs >= 0.10).astype(int)
        for nm, p in pred.items():
            p = np.nan_to_num(p, nan=np.nanmean(obs))
            r = dict(alvo=y, mes=str(m), modelo=nm, n=len(te), mae_MWh=np.sum(w * np.abs(p - obs)) / 1, obs_MWh=np.sum(w * obs), prev_MWh=np.sum(w * p),
                     brier=brier_score_loss(ev, np.clip(p * 2.5, 0, 1)) if False else np.mean((np.clip(p, 0, 1) - ev) ** 2),
                     auc=roc_auc_score(ev, p) if 0 < ev.sum() < len(ev) else np.nan, taxa_evento=ev.mean())
            res.append(r)
        print(y, m, flush=True)
R = pd.DataFrame(res); R.to_csv(os.path.join(OUT, 'previsao_mensal.csv'), index=False)
pd.concat(imp).to_csv(os.path.join(OUT, 'previsao_importancia.csv'), index=False)
S2 = R.groupby(['alvo', 'modelo']).agg(mae=('mae_MWh', 'sum'), obs=('obs_MWh', 'sum'), prev=('prev_MWh', 'sum'), auc=('auc', 'mean'), brier=('brier', 'mean')).reset_index()
for a in S2['alvo'].unique():
    m = S2['alvo'] == a
    S2.loc[m, 'skill_mae_vs_P'] = 1 - S2.loc[m, 'mae'] / S2.loc[m & (S2['modelo'] == 'P'), 'mae'].values[0]
    S2.loc[m, 'BSS_vs_C'] = 1 - S2.loc[m, 'brier'] / S2.loc[m & (S2['modelo'] == 'C'), 'brier'].values[0]
S2['mae_rel_obs'] = S2['mae'] / S2['obs']
S2.to_csv(os.path.join(OUT, 'previsao_resumo.csv'), index=False)
print(S2.round(3).to_string())
