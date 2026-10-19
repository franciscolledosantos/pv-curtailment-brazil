"""Análises complementares que geram números citados no texto (revisão de 17/10/2026).

Lê dados/ons_pv/*.parquet, dados/painel_pv_horario.parquet, dados/painel_modelo.parquet,
dados/previsoes_detalhe.parquet e resultados/*.csv. Grava em resultados/ e figuras/:
  razao_origem_GWh.csv            tabela razão × origem (REL/CNF/ENE × LOC/SIS)
  painel_entradas_saidas.csv      entradas e saídas de identificadores por ano; painel_ids_historico.csv por usina
  sensibilidade_painel_balanceado.csv  totais por ano/razão e regressão diária com as 56 usinas presentes em abr/2024
  ref_media7.csv                  referência média de 7 dias (mesma hora), com recuo para D-1 quando faltam dias
  bootstrap_skill.csv             IC 95% da habilidade (reamostragem dos 18 meses, 2.000 réplicas, semente 0)
  reg_diaria_logit_frac.csv       logit fracionário (efeitos marginais médios) e VIF
  reg_diaria_sensibilidade.csv    OLS com HAC 7/14/30 defasagens, com dummies de mês e modelo em duas partes
  concentracao.csv                Gini (energia e fração), parcela das 10 usinas mais afetadas
  faixas_carga_liquida.csv        fração ENE média por faixa de carga líquida ao meio-dia
  descricao_classes.csv           classificação das descrições (desde set/2025) por elemento de rede
  rel_horas_limiar.csv            horas REL por usina-ano a partir dos minutos publicados; limiar de 30,5 h
  auc_limiares.csv                AUC do evento "corte ≥ x" para x = 5, 10, 20% (alvos total e ENE; L vs P)
  robustez_balanceado_avaliacao.csv  habilidade do modelo completo avaliada só nas 56 usinas do painel balanceado
  reclassificacao_cnf.csv         CNF mensal (% do potencial) ago/2025 vs 2026
  figuras/Fig4_skill.png/.tif     habilidade mensal por alvo
"""
import glob, os
import numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
R = os.path.join(os.path.dirname(__file__), '..'); B = os.path.join(R, 'dados'); OUT = os.path.join(R, 'resultados'); F = os.path.join(R, 'figuras')

# ---------------------------------------------------------------- razão × origem (dados brutos, sem duplicatas)
pv = pd.concat([pd.read_parquet(f, columns=['id_ons', 'din_instante', 'cod_razaorestricao', 'cod_origemrestricao', 'val_geracaonaorealizadaapurada',
                                            'val_geracao', 'num_minutos_rel', 'dsc_restricao'])
                for f in sorted(glob.glob(os.path.join(B, 'ons_pv', '*.parquet')))], ignore_index=True)
pv = pv.drop_duplicates(['id_ons', 'din_instante'])
for c in ['cod_razaorestricao', 'cod_origemrestricao', 'dsc_restricao']: pv[c] = pv[c].astype(object)
pv['gnr'] = pd.to_numeric(pv['val_geracaonaorealizadaapurada'], errors='coerce').fillna(0).clip(lower=0) / 2  # MWmed/meia hora -> MWh
ro = pv[pv['gnr'] > 0].pivot_table(index='cod_razaorestricao', columns='cod_origemrestricao', values='gnr', aggfunc='sum', fill_value=0) / 1e3
ro['total'] = ro.sum(axis=1); ro['pct_SIS'] = 100 * ro.get('SIS', 0) / ro['total']; ro['pct_LOC'] = 100 * ro.get('LOC', 0) / ro['total']
ro.loc['total'] = ro.sum(); ro.loc['total', 'pct_SIS'] = 100 * ro.loc['total', 'SIS'] / ro.loc['total', 'total']; ro.loc['total', 'pct_LOC'] = 100 - ro.loc['total', 'pct_SIS']
ro.round(2).to_csv(os.path.join(OUT, 'razao_origem_GWh.csv')); print('razão × origem (GWh)\n', ro.round(1))
print('códigos de razão presentes:', sorted(pv['cod_razaorestricao'].dropna().unique()))

# ---------------------------------------------------------------- limiar de 30,5 h (minutos REL publicados)
pv['h_rel'] = pd.to_numeric(pv['num_minutos_rel'], errors='coerce').fillna(0) / 60
pv['gnr_rel'] = np.where(pv['cod_razaorestricao'] == 'REL', pv['gnr'], 0.0)
rl = pv.assign(ano=pv['din_instante'].dt.year).groupby(['id_ons', 'ano']).agg(h_rel=('h_rel', 'sum'), gnr_rel=('gnr_rel', 'sum')).reset_index()
rl['acima'] = rl['h_rel'] > 30.5; rl.to_csv(os.path.join(OUT, 'rel_horas_limiar.csv'), index=False)
print('REL acima de 30,5 h: %.1f%% da energia REL; %d de %d usina-anos' % (100 * rl.loc[rl['acima'], 'gnr_rel'].sum() / rl['gnr_rel'].sum(), rl['acima'].sum(), len(rl)))

# ---------------------------------------------------------------- descrições por elemento de rede (desde set/2025)
d = pv[(pv['gnr'] > 0) & pv['cod_razaorestricao'].isin(['REL', 'CNF']) & (pv['din_instante'] >= '2025-09-01')]
dd = d.groupby('dsc_restricao', dropna=False)['gnr'].sum().div(1e3).sort_values(ascending=False).reset_index()
def classe(s):
    s = str(s).lower()
    if s in ('nan', 'none', ''): return 'sem descrição'
    if ('conjunto' in s or 'ufv' in s) and not any(k in s for k in ['transforma', ' lt ', 'lts ', 'fnese', 'fjusc', 'bipolo']): return 'other'  # limites de geração de conjuntos
    if 'transforma' in s or ' tr ' in s or 'tr1' in s or 'tr9' in s or 'tr-' in s or 'atr' in s: return 'transformation'
    if 'intercâmbio' in s or 'intercambio' in s or 'fnese' in s or 'fluxo' in s or 'bipolo' in s or 'xingu' in s or 'exportação' in s or 'exportacao' in s or 'recebimento' in s or 'fsul' in s or 'fne' in s or 'fjusc' in s: return 'inter-regional flow'
    if 'lt ' in s or 'linha' in s or ' kv ' in s: return 'transmission line'
    return 'other'
dd['classe'] = dd['dsc_restricao'].map(classe); dd['rio_novo_do_sul'] = dd['dsc_restricao'].astype(str).str.lower().str.contains('rio novo do sul')
dd.to_csv(os.path.join(OUT, 'descricao_classes.csv'), index=False)
cl = dd.groupby('classe')['gnr'].sum(); cl = pd.DataFrame({'GWh': cl, 'pct': 100 * cl / cl.sum()}); print('classes (REL+CNF desde set/2025, %.0f GWh)\n' % cl['GWh'].sum(), cl.round(1))
rns = dd.loc[dd['rio_novo_do_sul'] & (dd['classe'] == 'transformation'), 'gnr'].sum(); print('Rio Novo do Sul (transformação): %.0f GWh = %.1f%%' % (rns, 100 * rns / cl['GWh'].sum()))
cl.round(2).to_csv(os.path.join(OUT, 'descricao_classes_resumo.csv'))
ene_d = pv[(pv['gnr'] > 0) & (pv['cod_razaorestricao'] == 'ENE') & (pv['din_instante'] >= '2025-09-01')].groupby('dsc_restricao', dropna=False)['gnr'].sum().div(1e3)
print('ENE desde set/2025 por descrição (GWh)\n', ene_d.round(1))

# ---------------------------------------------------------------- painel: entradas e saídas
H = pd.read_parquet(os.path.join(B, 'painel_pv_horario.parquet'))
meta = pd.read_csv(os.path.join(OUT, 'usinas_pv.csv'), index_col=0)
hist = H[H['pot'] > 0].groupby('id_ons')['hora'].agg(['min', 'max']).join(meta[['nom_usina', 'uf', 'ponto', 'cap']])
hist['entrada_tardia'] = hist['min'] >= '2024-05-01'; hist['saida_precoce'] = hist['max'] < '2026-09-01'
hist.to_csv(os.path.join(OUT, 'painel_ids_historico.csv'))
es = pd.DataFrame({'entradas (1º registro no ano)': hist.groupby(hist['min'].dt.year).size(), 'saídas (último registro no ano)': hist[hist['saida_precoce']].groupby(hist['max'].dt.year).size()}).fillna(0).astype(int)
es.to_csv(os.path.join(OUT, 'painel_entradas_saidas.csv')); print(es)
print('ids presentes em abr/2024:', (~hist['entrada_tardia']).sum(), '; entradas depois:', hist['entrada_tardia'].sum(), '; saídas antes de set/2026:', hist['saida_precoce'].sum())
bal = hist.index[~hist['entrada_tardia']].tolist()

# ---------------------------------------------------------------- painel balanceado: totais e regressão
COLS = ['gnr', 'pot', 'gnr_REL', 'gnr_CNF', 'gnr_ENE']
Hb = H[H['id_ons'].isin(bal)]
Yb = Hb.assign(ano=Hb['hora'].dt.year).groupby('ano')[COLS].sum() / 1e3
Yb['pct'] = 100 * Yb['gnr'] / Yb['pot']; Yb['pct_ENE_do_corte'] = 100 * Yb['gnr_ENE'] / Yb['gnr']; Yb['pct_grid_do_corte'] = 100 * (Yb['gnr_REL'] + Yb['gnr_CNF']) / Yb['gnr']
Yb.loc['total'] = Yb.sum(); Yb.loc['total', 'pct'] = 100 * Yb.loc['total', 'gnr'] / Yb.loc['total', 'pot']
Yb.loc['total', 'pct_ENE_do_corte'] = 100 * Yb.loc['total', 'gnr_ENE'] / Yb.loc['total', 'gnr']; Yb.loc['total', 'pct_grid_do_corte'] = 100 - Yb.loc['total', 'pct_ENE_do_corte']
print('painel balanceado (56 ids)\n', Yb.round(1))
S = pd.read_parquet(os.path.join(B, 'sistema_horario.parquet'))
def diario(Hx):
    Hx = Hx.assign(h=Hx['hora'].dt.hour); d = Hx[Hx['h'].between(10, 15)].groupby(Hx['hora'].dt.normalize())[['gnr_ENE', 'gnr_REL', 'gnr_CNF', 'pot']].sum()
    Sm = S[S.index.hour.isin(range(10, 16))]; s = Sm.groupby(Sm.index.normalize())[['carga_SIN', 'eol_SIN', 'pv_SIN', 'liq_SIN']].mean()
    d = d.join(s, how='inner'); d['f_ene'] = d['gnr_ENE'] / d['pot']; d['f_grid'] = (d['gnr_REL'] + d['gnr_CNF']) / d['pot']
    d['fds'] = (d.index.dayofweek >= 5).astype(int); d['t'] = (d.index - d.index.min()).days / 365.25
    for c in ['liq', 'eol', 'carga', 'pv']: d[c + '_GW'] = d[c + '_SIN'] / 1e3
    d['mes'] = d.index.month; return d
db = diario(Hb)
mb = smf.ols('f_ene ~ carga_GW + eol_GW + pv_GW + fds + t', db).fit(cov_type='HAC', cov_kwds={'maxlags': 7})
rb = pd.DataFrame({'coef': mb.params, 'se': mb.bse, 'p': mb.pvalues}); rb.loc['R2', 'coef'] = mb.rsquared
rb.to_csv(os.path.join(OUT, 'sensibilidade_painel_balanceado_regressao.csv')); Yb.round(2).to_csv(os.path.join(OUT, 'sensibilidade_painel_balanceado.csv'))
print('regressão ENE, painel balanceado\n', rb.round(4))

# ---------------------------------------------------------------- regressão diária: logit fracionário, VIF, sensibilidade
d = pd.read_csv(os.path.join(OUT, 'diario_sistema.csv'), index_col=0, parse_dates=True); d['mes'] = d.index.month
X = sm.add_constant(d[['carga_GW', 'eol_GW', 'pv_GW', 'fds', 't']])
vif = pd.Series({c: variance_inflation_factor(X.values, i) for i, c in enumerate(X.columns) if c != 'const'})
rows = []
for y in ['f_ene', 'f_grid']:
    gl = sm.GLM(d[y], X, family=sm.families.Binomial()).fit(cov_type='HAC', cov_kwds={'maxlags': 7})
    ame = gl.get_margeff(at='overall').summary_frame()
    for v in ['carga_GW', 'eol_GW', 'pv_GW', 'fds', 't']:
        rows.append(dict(alvo=y, var=v, AME=ame.loc[v, 'dy/dx'], p=ame.loc[v, 'Pr(>|z|)'], VIF=vif[v]))
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'reg_diaria_logit_frac.csv'), index=False); print('logit fracionário\n', pd.DataFrame(rows).round(4))
sens = []
for y in ['f_ene', 'f_grid']:
    for lag in [7, 14, 30]:
        m = smf.ols(f'{y} ~ carga_GW + eol_GW + pv_GW + fds + t', d).fit(cov_type='HAC', cov_kwds={'maxlags': lag})
        for v in ['carga_GW', 'eol_GW', 'pv_GW', 'fds', 't']: sens.append(dict(alvo=y, modelo=f'OLS HAC({lag})', var=v, coef=m.params[v], se=m.bse[v], p=m.pvalues[v], R2=m.rsquared))
    m = smf.ols(f'{y} ~ carga_GW + eol_GW + pv_GW + fds + t + C(mes)', d).fit(cov_type='HAC', cov_kwds={'maxlags': 14})
    for v in ['carga_GW', 'eol_GW', 'pv_GW', 'fds', 't']: sens.append(dict(alvo=y, modelo='OLS HAC(14) + dummies de mês', var=v, coef=m.params[v], se=m.bse[v], p=m.pvalues[v], R2=m.rsquared))
# modelo em duas partes para ENE: ocorrência (logit) × intensidade (OLS nos dias com corte)
d['ene_pos'] = (d['f_ene'] > 0).astype(int)
lg = smf.logit('ene_pos ~ carga_GW + eol_GW + pv_GW + fds + t', d).fit(disp=0)
am = lg.get_margeff().summary_frame()
for v in ['carga_GW', 'eol_GW', 'pv_GW', 'fds', 't']: sens.append(dict(alvo='f_ene', modelo='duas partes: ocorrência (logit, AME)', var=v, coef=am.loc[v, 'dy/dx'], se=am.loc[v, 'Std. Err.'], p=am.loc[v, 'Pr(>|z|)'], R2=lg.prsquared))
mi = smf.ols('f_ene ~ carga_GW + eol_GW + pv_GW + fds + t', d[d['ene_pos'] == 1]).fit(cov_type='HAC', cov_kwds={'maxlags': 14})
for v in ['carga_GW', 'eol_GW', 'pv_GW', 'fds', 't']: sens.append(dict(alvo='f_ene', modelo='duas partes: intensidade (OLS, dias com ENE>0)', var=v, coef=mi.params[v], se=mi.bse[v], p=mi.pvalues[v], R2=mi.rsquared))
sens = pd.DataFrame(sens); sens.to_csv(os.path.join(OUT, 'reg_diaria_sensibilidade.csv'), index=False); print(sens.round(4).to_string())
print('dias com ENE > 0: %d de %d (%.0f%%)' % (d['ene_pos'].sum(), len(d), 100 * d['ene_pos'].mean()))

# ---------------------------------------------------------------- faixas de carga líquida
fx = pd.cut(d['liq_GW'], [0, 40, 50, 65, 80, 200], labels=['<40', '40–50', '50–65', '65–80', '>80'])
fc = d.groupby(fx).agg(dias=('f_ene', 'size'), f_ene_media=('f_ene', 'mean'), dias_com_ENE=('ene_pos', 'sum')); fc.to_csv(os.path.join(OUT, 'faixas_carga_liquida.csv')); print(fc.round(3))

# ---------------------------------------------------------------- concentração
P = pd.read_csv(os.path.join(OUT, 'por_usina.csv'), index_col=0)
def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x); return (2 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum())) - (n + 1) / n
top = P['grid'].nlargest(10).index
conc = pd.Series({'gini_energia_rede': gini(P['grid']), 'gini_energia_ENE': gini(P['gnr_ENE']), 'gini_fracao_rede': gini(P['pct_grid']),
                  'top10_parcela_corte_rede': P.loc[top, 'grid'].sum() / P['grid'].sum(), 'top10_parcela_potencial': P.loc[top, 'pot'].sum() / P['pot'].sum()})
conc.to_csv(os.path.join(OUT, 'concentracao.csv')); print(conc.round(3))

# ---------------------------------------------------------------- previsão: média de 7 dias, bootstrap, AUC por limiar, painel balanceado
G = pd.read_parquet(os.path.join(B, 'painel_modelo.parquet'), columns=['id_ons', 'hora', 'pot', 'f', 'f_grid', 'f_ene', 'f_l24', 'f_grid_l24', 'f_ene_l24', 'f_m7', 'f_grid_m7', 'f_ene_m7'])
G['ym'] = G['hora'].dt.to_period('M'); G = G[(G['ym'] >= '2025-04') & (G['ym'] <= '2026-09')]
m7 = []
for y in ['f', 'f_grid', 'f_ene']:
    te = G[G[y].notna() & G[y + '_l24'].notna()]
    p7 = te[y + '_m7'].fillna(te[y + '_l24'])  # recuo para D-1 quando faltam dias na janela
    e7 = (te['pot'] * (p7 - te[y]).abs()).sum(); eP = (te['pot'] * (te[y + '_l24'] - te[y]).abs()).sum()
    m7.append(dict(alvo=y, modelo='M7', mae=e7, obs=(te['pot'] * te[y]).sum(), skill_vs_P=1 - e7 / eP, mae_rel_obs=e7 / (te['pot'] * te[y]).sum()))
m7 = pd.DataFrame(m7); m7.to_csv(os.path.join(OUT, 'ref_media7.csv'), index=False); print(m7.round(3))
Rm = pd.read_csv(os.path.join(OUT, 'previsao_mensal.csv')); rng = np.random.default_rng(0); bs = []
for y in ['f', 'f_ene', 'f_grid']:
    for mdl in ['L', 'LO']:
        a = Rm[(Rm['alvo'] == y) & (Rm['modelo'] == mdl)].set_index('mes')['mae_MWh']; p = Rm[(Rm['alvo'] == y) & (Rm['modelo'] == 'P')].set_index('mes')['mae_MWh']
        meses = a.index.values; sk = []
        for _ in range(2000):
            i = rng.integers(0, len(meses), len(meses)); sk.append(1 - a.values[i].sum() / p.values[i].sum())
        bs.append(dict(alvo=y, modelo=mdl, skill=1 - a.sum() / p.sum(), ic95_inf=np.percentile(sk, 2.5), ic95_sup=np.percentile(sk, 97.5), meses_melhor=int((a < p).sum())))
bs = pd.DataFrame(bs); bs.to_csv(os.path.join(OUT, 'bootstrap_skill.csv'), index=False); print(bs.round(3))
Pd = pd.read_parquet(os.path.join(B, 'previsoes_detalhe.parquet')); Pd['ym'] = Pd['hora'].dt.to_period('M'); au = []
for y in ['f', 'f_ene']:
    for thr in [0.05, 0.10, 0.20]:
        for mdl in ['p_L', 'p_P']:
            v = []
            for m, g in Pd[Pd['alvo'] == y].groupby('ym'):
                ev = (g['obs'] >= thr).astype(int)
                if 0 < ev.sum() < len(ev): v.append(roc_auc_score(ev, g[mdl]))
            au.append(dict(alvo=y, limiar=thr, modelo=mdl[2:], auc=np.mean(v)))
au = pd.DataFrame(au); au.to_csv(os.path.join(OUT, 'auc_limiares.csv'), index=False); print(au.round(3))
def sk(g):
    e1 = (g['pot'] * (g['p_L'] - g['obs']).abs()).sum(); e0 = (g['pot'] * (g['p_P'] - g['obs']).abs()).sum(); return pd.Series({'skill': 1 - e1 / e0, 'curt_GWh': (g['pot'] * g['obs']).sum() / 1e3, 'n': len(g)})
Pd['painel'] = np.where(Pd['id_ons'].isin(bal), 'balanceado (56 ids)', 'entradas tardias (34 ids)')
rb2 = Pd.groupby(['alvo', 'painel']).apply(sk); rb2.to_csv(os.path.join(OUT, 'robustez_balanceado_avaliacao.csv')); print(rb2.round(3))

# ---------------------------------------------------------------- CNF: reclassificação em 2026
M = pd.read_csv(os.path.join(OUT, 'mensal.csv'), index_col=0)
rc = M.loc[['2025-08', '2025-09', '2025-10'] + [m for m in M.index if m >= '2026-01'], ['pct_gnr_CNF', 'pct_gnr_REL', 'pct_gnr_ENE']]
rc.to_csv(os.path.join(OUT, 'reclassificacao_cnf.csv')); print(rc.round(1))

# ---------------------------------------------------------------- Fig. 4: habilidade mensal
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
for ax, (y, tt) in zip(axs, [('f', '(a) Total'), ('f_ene', '(b) Energy surplus'), ('f_grid', '(c) Grid (REL + CNF)')]):
    r = Rm[Rm['alvo'] == y].pivot(index='mes', columns='modelo', values='mae_MWh')
    x = np.arange(len(r)); w = 0.27
    for k, (mdl, lab, col) in enumerate([('C', 'Climatology', '#999999'), ('L', 'LightGBM, D−1 data', '#2a6fb0'), ('LO', 'LightGBM + perfect system forecast', '#e6a100')]):
        s = 100 * (1 - r[mdl] / r['P']); s = s.where(s > -100)
        ax.bar(x + (k - 1) * w, s, width=w, color=col, label=lab)
    ax.axhline(0, color='k', lw=0.6); ax.set_xticks(x[::3]); ax.set_xticklabels(r.index[::3], rotation=90, fontsize=6); ax.set_title(tt, loc='left', fontsize=8)
axs[0].set_ylabel('Skill vs persistence (%)'); axs[0].set_ylim(-100, 60); axs[1].legend(frameon=False, fontsize=6, loc='lower left')
plt.tight_layout(); [plt.savefig(os.path.join(F, f'Fig4_skill.{e}'), dpi=300) for e in ('png', 'tif')]; plt.close()
print('ok')
