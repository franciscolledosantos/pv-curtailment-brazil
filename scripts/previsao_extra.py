"""Experimentos adicionais de previsão D+1 pedidos na revisão de 17/10/2026.

Reutiliza o painel de modelagem de `previsao.py` (dados/painel_modelo.parquet) e o mesmo desenho de
validação (janela expansível mensal, abr/2025–set/2026; teste restrito às usina-horas com D-1 válido).
Seções (cada uma grava seus próprios arquivos em resultados/); uso: python previsao_extra.py 3 2 (ordem livre):
  1. alvos "rede local" (LOC) e "rede sistêmica" (REL+CNF de origem SIS): P, C, L
  2. recalibração em janela recente, alvo ENE (o mais sub-predito): L treinado só nos últimos 12 meses
     e L com ponderação por recência (meia-vida de 6 meses) sobre toda a história
  3. cenário do terceiro com dados defasados de 2 dias (alvo total): todas as defasagens recuadas em 24 h
  4. referência linear: logit fracionário (GLM binomial) com as mesmas defasagens e calendário (três alvos),
     ajustado em subamostra aleatória de até 250 mil usina-horas do treino (modelo linear; custo de computação);
     ajuste por IRLS em numpy (equivalente ao GLM binomial com var_weights)
  5. capacidade calculada só com o passado (p95 expansível), alvo total; lista das 56 usinas do painel
     balanceado (a habilidade avaliada só nessas usinas está em analises_complementares.py)
  6. importância por permutação (erro em energia) no modelo final de set/2026 (alvos total e rede)
Hiperparâmetros do LightGBM fixados a priori, iguais aos de previsao.py. Por custo de computação
(2 núcleos compartilhados; ~2 min por ajuste), as seções 2, 3 e 5 foram restritas aos alvos indicados.
"""
import os, sys, warnings, time
import numpy as np, pandas as pd, lightgbm as lgb
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
warnings.filterwarnings('ignore')
B = os.path.join(os.path.dirname(__file__), '..', 'dados')
OUT = os.path.join(os.path.dirname(__file__), '..', 'resultados')
SEC = sys.argv[1:] or ['1', '2', '3', '4', '5', '6']
t0 = time.time()
def log(*a): print(f'[{(time.time()-t0)/60:5.1f} min]', *a, flush=True)

_SYS = ['carga_SIN', 'carga_NE', 'carga_SE', 'eol_SIN', 'eol_NE', 'pv_SIN', 'liq_SIN', 'ene_frac_SIN']
if set(SEC) <= {'4'}:  # seção 4 sozinha: só as colunas necessárias (memória)
    _c = ['id_ons', 'uf', 'sub', 'ponto', 'hora', 'pot', 'h', 'dow', 'mes', 'feriado', 'doy_s', 'doy_c', 'cap', 'pot_l24', 'pot_m7'] + \
         [y + s for y in ['f', 'f_grid', 'f_ene'] for s in ['', '_l24', '_l48', '_l168', '_m7', '_d1']] + [c + '_l24' for c in _SYS]
    G = pd.read_parquet(os.path.join(B, 'painel_modelo.parquet'), columns=_c)
else:
    G = pd.read_parquet(os.path.join(B, 'painel_modelo.parquet'))
G = G.drop(columns=[c for c in ['ger', 'gnr', 'gnr_REL', 'gnr_CNF', 'gnr_ENE', 'gnr_LOC', 'lim', 'val_disponibilidade', 'pot_l48', 'pot_l168',
                                'carga_SIN', 'carga_NE', 'eol_SIN', 'eol_NE', 'pv_SIN', 'liq_SIN'] if c in G.columns])
for c in G.select_dtypes('float64').columns: G[c] = G[c].astype('float32')  # memória (máquina compartilhada)
for c in ['id_ons', 'uf', 'sub', 'ponto']:
    G[c + '_cat'] = G[c].astype('category')
import gc; gc.collect()
G['ym'] = G['hora'].dt.to_period('M')
BASE = ['h', 'dow', 'mes', 'feriado', 'doy_s', 'doy_c', 'cap', 'id_ons_cat', 'uf_cat', 'sub_cat', 'ponto_cat', 'pot_l24', 'pot_m7']
OTHER = ['f_l24', 'f_grid_l24', 'f_ene_l24', 'f_m7', 'f_grid_m7', 'f_ene_m7']
LAGS = lambda y: [y + '_l24', y + '_l48', y + '_l168', y + '_m7', y + '_d1'] + [z for z in OTHER if not z.startswith(y + '_')]
SYSV = ['carga_SIN', 'carga_NE', 'carga_SE', 'eol_SIN', 'eol_NE', 'pv_SIN', 'liq_SIN', 'ene_frac_SIN']
SYS1 = [c + '_l24' for c in SYSV]
MESES = pd.period_range('2025-04', '2026-09', freq='M')
if os.environ.get('TESTE'): MESES = MESES[-2:]
PAR = dict(objective='cross_entropy', n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=100,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)

def fit_lgb(tr, fs, y):
    m = lgb.LGBMRegressor(**PAR); m.fit(tr[fs], tr[y], sample_weight=tr['pot'].clip(lower=0.5)); return m

def metrics(te, y, preds, alvo, mes, extra=None):
    w = te['pot'].values; obs = te[y].values; ev = (obs >= 0.10).astype(int); rows = []
    for nm, p in preds.items():
        p = np.nan_to_num(np.asarray(p, float), nan=np.nanmean(obs))
        rows.append(dict(alvo=alvo, mes=str(mes), modelo=nm, n=len(te), mae_MWh=np.sum(w * np.abs(p - obs)), obs_MWh=np.sum(w * obs),
                         prev_MWh=np.sum(w * p), auc=roc_auc_score(ev, p) if 0 < ev.sum() < len(ev) else np.nan, **(extra or {})))
    return rows

def resumo(R, ref='P'):
    S = R.groupby(['alvo', 'modelo']).agg(mae=('mae_MWh', 'sum'), obs=('obs_MWh', 'sum'), prev=('prev_MWh', 'sum'), auc=('auc', 'mean'), n=('n', 'sum')).reset_index()
    for a in S['alvo'].unique():
        m = S['alvo'] == a
        S.loc[m, 'skill_vs_' + ref] = 1 - S.loc[m, 'mae'] / S.loc[m & (S['modelo'] == ref), 'mae'].values[0]
    S['mae_rel_obs'] = S['mae'] / S['obs']; S['bias'] = S['prev'] / S['obs'] - 1
    return S

def clim(tr, te, y):
    c1 = tr.groupby(['id_ons', 'h', 'mes'])[y].mean(); c2 = tr.groupby(['id_ons', 'h'])[y].mean()
    c = pd.Series(list(zip(te['id_ons'], te['h'], te['mes']))).map(c1)
    return c.fillna(pd.Series(list(zip(te['id_ons'], te['h']))).map(c2)).fillna(tr[y].mean()).values

def lags_completos(H, cols, extra_shift=0):
    """Defasagens por usina sobre calendário horário completo (como em previsao.py).
    extra_shift=24 recua tudo um dia (cenário D-2)."""
    Hs = H.set_index(['id_ons', 'hora']).sort_index()
    idx = pd.MultiIndex.from_product([Hs.index.levels[0], pd.date_range(Hs.index.get_level_values(1).min(), Hs.index.get_level_values(1).max(), freq='h')], names=['id_ons', 'hora'])
    X = Hs[cols].reindex(idx); out = pd.DataFrame(index=idx); s = extra_shift
    for c in cols:
        g = X[c].groupby(level=0)
        out[c + '_l24'] = g.shift(24 + s); out[c + '_l48'] = g.shift(48 + s); out[c + '_l168'] = g.shift(168 + s)
        out[c + '_m7'] = sum(g.shift(24 * k + s) for k in range(1, 8)) / 7
    out['dia'] = out.index.get_level_values(1).normalize()
    dm = X.join(out[['dia']]).groupby([X.index.get_level_values(0), 'dia'])[cols].mean()
    dm = dm.groupby(level=0).shift(1 + s // 24).add_suffix('_d1')
    keep = X[cols[0]].notna().values | X[cols[-1]].notna().values
    out = out[keep].astype({c: 'float32' for c in out.columns if c != 'dia'})
    out = out.reset_index().merge(dm.astype('float32').reset_index().rename(columns={'level_0': 'id_ons'}), on=['id_ons', 'dia'], how='left').drop(columns='dia')
    return out

# ---------------------------------------------------------------- 1. alvos LOC e rede SIS
if '1' in SEC:
    log('1. alvos rede local (LOC) e rede sistêmica')
    H = pd.read_parquet(os.path.join(B, 'painel_pv_horario.parquet'))
    H['h'] = H['hora'].dt.hour; H = H[H['h'].between(6, 18)].copy()
    ok = H['pot'] > 0.5
    H['f_loc'] = np.where(ok, (H['gnr_LOC'] / H['pot']).clip(0, 1), np.nan)
    H['f_gsis'] = np.where(ok, ((H['gnr_REL'] + H['gnr_CNF'] - H['gnr_LOC']).clip(lower=0) / H['pot']).clip(0, 1), np.nan)
    L1 = lags_completos(H, ['f_loc', 'f_gsis'])
    G1 = G.merge(H[['id_ons', 'hora', 'f_loc', 'f_gsis']], on=['id_ons', 'hora'], how='left').merge(L1, on=['id_ons', 'hora'], how='left')
    res = []
    for y in ['f_loc', 'f_gsis']:
        fs = BASE + [y + '_l24', y + '_l48', y + '_l168', y + '_m7', y + '_d1'] + OTHER + SYS1
        for m in MESES:
            tr = G1[(G1['ym'] < m) & G1[y].notna()]; te = G1[(G1['ym'] == m) & G1[y].notna() & G1[y + '_l24'].notna()]
            if len(te) == 0: continue
            pred = {'P': te[y + '_l24'].values, 'C': clim(tr, te, y), 'L': np.clip(fit_lgb(tr, fs, y).predict(te[fs]), 0, 1)}
            res += metrics(te, y, pred, y, m)
        log(y, 'ok')
    R = pd.DataFrame(res); R.to_csv(os.path.join(OUT, 'previsao_loc_mensal.csv'), index=False)
    S = resumo(R); S.to_csv(os.path.join(OUT, 'previsao_loc_resumo.csv'), index=False); print(S.round(3).to_string(), flush=True)

# ---------------------------------------------------------------- 2. recalibração em janela recente
if '2' in SEC:
    log('2. recalibração em janela recente (12 meses e ponderação por recência), alvo ENE')
    res = []
    for y in ['f_ene']:
        fs = BASE + LAGS(y) + SYS1
        for m in MESES:
            te = G[(G['ym'] == m) & G[y].notna() & G[y + '_l24'].notna()]
            pred = {'P': te[y + '_l24'].values}
            for W in [12]:
                tr = G[(G['ym'] < m) & (G['ym'] >= m - W) & G[y].notna()]
                pred[f'L_{W}m'] = np.clip(fit_lgb(tr, fs, y).predict(te[fs]), 0, 1)
            # ponderação por recência (meia-vida de 6 meses) sobre toda a história
            tr = G[(G['ym'] < m) & G[y].notna()]
            idade = m.ordinal - pd.PeriodIndex(tr['ym']).asi8
            mdl = lgb.LGBMRegressor(**PAR); mdl.fit(tr[fs], tr[y], sample_weight=tr['pot'].clip(lower=0.5) * 0.5 ** (idade / 6))
            pred['L_rec6'] = np.clip(mdl.predict(te[fs]), 0, 1)
            res += metrics(te, y, pred, y, m)
            log(y, m)
    R = pd.DataFrame(res); R.to_csv(os.path.join(OUT, 'previsao_recal_mensal.csv'), index=False)
    S = resumo(R); S.to_csv(os.path.join(OUT, 'previsao_recal_resumo.csv'), index=False); print(S.round(3).to_string(), flush=True)

# ---------------------------------------------------------------- 3. dados defasados de 2 dias
if '3' in SEC:
    log('3. cenário D-2 (dados abertos com atraso de um dia)')
    H = pd.read_parquet(os.path.join(B, 'painel_pv_horario.parquet'))
    H['h'] = H['hora'].dt.hour; H = H[H['h'].between(6, 18)].copy(); ok = H['pot'] > 0.5
    H['f'] = np.where(ok, (H['gnr'] / H['pot']).clip(0, 1), np.nan)
    H['f_grid'] = np.where(ok, ((H['gnr_REL'] + H['gnr_CNF']) / H['pot']).clip(0, 1), np.nan)
    H['f_ene'] = np.where(ok, (H['gnr_ENE'] / H['pot']).clip(0, 1), np.nan)
    H['potv'] = H['pot']
    L2 = lags_completos(H, ['f', 'f_grid', 'f_ene', 'potv'], extra_shift=24)
    L2.columns = [c + '_D2' if c not in ('id_ons', 'hora') else c for c in L2.columns]
    G2 = G.merge(L2, on=['id_ons', 'hora'], how='left')
    S = pd.read_parquet(os.path.join(B, 'sistema_horario.parquet'))
    pvsys = H.groupby('hora')[['gnr_ENE', 'pot']].sum(); S['ene_frac_SIN'] = pvsys['gnr_ENE'] / pvsys['pot']
    for c in SYSV: S[c + '_l48'] = S[c].shift(48)
    G2 = G2.merge(S[[c + '_l48' for c in SYSV]], left_on='hora', right_index=True, how='left')
    res = []
    for y in ['f']:
        fs = ['h', 'dow', 'mes', 'feriado', 'doy_s', 'doy_c', 'cap', 'id_ons_cat', 'uf_cat', 'sub_cat', 'ponto_cat', 'potv_l24_D2', 'potv_m7_D2'] + \
             [y + s + '_D2' for s in ['_l24', '_l48', '_l168', '_m7', '_d1']] + [z + '_D2' for z in OTHER if not z.startswith(y + '_')] + [c + '_l48' for c in SYSV]
        for m in MESES:
            tr = G2[(G2['ym'] < m) & G2[y].notna()]
            te = G2[(G2['ym'] == m) & G2[y].notna() & G2[y + '_l24'].notna()]
            pred = {'P': te[y + '_l24'].values, 'P_D2': te[y + '_l24_D2'].values, 'L_D2': np.clip(fit_lgb(tr, fs, y).predict(te[fs]), 0, 1)}
            res += metrics(te, y, pred, y, m)
        log(y, 'ok')
    R = pd.DataFrame(res); R.to_csv(os.path.join(OUT, 'previsao_d2_mensal.csv'), index=False)
    S2 = resumo(R); S2.to_csv(os.path.join(OUT, 'previsao_d2_resumo.csv'), index=False); print(S2.round(3).to_string(), flush=True)

# ---------------------------------------------------------------- 4. referência linear (logit fracionário)
def irls_logit_frac(X, y, w, maxiter=50, tol=1e-8, ridge=1e-6):
    """Logit fracionário (quase-verossimilhança binomial, Papke & Wooldridge) por IRLS/Newton em numpy, com pesos w.
    Equivale a sm.GLM(..., family=Binomial(), var_weights=w); escrito à mão porque o statsmodels esgotava a memória."""
    beta = np.zeros(X.shape[1])
    for _ in range(maxiter):
        mu = 1 / (1 + np.exp(-np.clip(X @ beta, -30, 30))); v = w * mu * (1 - mu)
        H = X.T @ (X * v[:, None]) + ridge * np.eye(X.shape[1]); g = X.T @ (w * (y - mu))
        step = np.linalg.solve(H, g); beta += step
        if np.max(np.abs(step)) < tol: break
    return beta

if '4' in SEC:
    log('4. referência linear: logit fracionário com as mesmas defasagens')
    res = []
    for y in ['f', 'f_ene', 'f_grid']:
        num = ['feriado', 'doy_s', 'doy_c', 'cap', 'pot_l24', 'pot_m7'] + LAGS(y) + SYS1
        for m in MESES:
            cols = list(dict.fromkeys(num + ['id_ons', 'h', 'dow', 'pot', y]))
            itr = G.index[(G['ym'] < m) & G[y].notna()]
            if len(itr) > 250_000: itr = pd.Index(np.random.default_rng(0).choice(itr, 250_000, replace=False))  # subamostra (memória/tempo; modelo linear)
            tr = G.loc[itr, cols]; te = G.loc[(G['ym'] == m) & G[y].notna() & G[y + '_l24'].notna(), cols]
            med = tr[num].median()
            umean = tr.groupby('id_ons')[y].mean()  # identidade da usina como média de treino (evita 90 dummies)
            def X(d):
                x = d[num].copy(); miss = x[[y + '_l168', y + '_m7', y + '_d1']].isna().astype(float).add_suffix('_na')
                x = x.fillna(med); x['u_mean'] = d['id_ons'].map(umean).fillna(tr[y].mean()).values
                x = pd.concat([x, miss, pd.get_dummies(d['h'], prefix='h', drop_first=True, dtype=float),
                               pd.get_dummies(d['dow'], prefix='dow', drop_first=True, dtype=float)], axis=1)
                return x
            Xtr, Xte = X(tr), X(te); Xte = Xte.reindex(columns=Xtr.columns, fill_value=0.0)
            mu, sd = Xtr.mean(), Xtr.std().replace(0, 1); Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
            try:
                beta = irls_logit_frac(np.c_[np.ones(len(Xtr)), Xtr.values], tr[y].values.astype(float), tr['pot'].clip(lower=0.5).values.astype(float))
                p = np.clip(1 / (1 + np.exp(-(np.c_[np.ones(len(Xte)), Xte.values] @ beta))), 0, 1)
            except Exception as e:
                log('GLM falhou', y, m, e); p = te[y + '_l24'].values
            res += metrics(te, y, {'P': te[y + '_l24'].values, 'GLM': p}, y, m)
            log(y, m)
    R = pd.DataFrame(res); R.to_csv(os.path.join(OUT, 'previsao_linear_mensal.csv'), index=False)
    S = resumo(R); S.to_csv(os.path.join(OUT, 'previsao_linear_resumo.csv'), index=False); print(S.round(3).to_string(), flush=True)

# ---------------------------------------------------------------- 5. painel balanceado e capacidade expansível
if '5' in SEC:
    log('5. cap expansível (p95 só com o passado) e lista do painel balanceado')
    H = pd.read_parquet(os.path.join(B, 'painel_pv_horario.parquet'))
    bal = sorted(H[(H['hora'] < '2024-05-01') & (H['pot'] > 0)]['id_ons'].unique())
    # p95 expansível da disponibilidade: por usina, quantil 0,95 das horas até o dia D-1
    H['dia'] = H['hora'].dt.normalize()
    cap_exp = []
    for u, g in H.sort_values('hora').groupby('id_ons'):
        v = g['val_disponibilidade'].values; d = g['dia'].values
        dias = pd.Series(d).drop_duplicates().values; q = []
        for dd in dias:
            q.append(np.nanquantile(v[d < dd], 0.95) if (d < dd).sum() > 0 else np.nan)
        cap_exp.append(pd.DataFrame({'id_ons': u, 'dia': dias, 'cap_exp': q}))
    cap_exp = pd.concat(cap_exp)
    G5 = G.merge(cap_exp, on=['id_ons', 'dia'], how='left')
    pd.Series(bal, name='id_ons').to_csv(os.path.join(OUT, 'painel_balanceado_ids.csv'), index=False)
    res = []
    y = 'f'; fs = BASE + LAGS(y) + SYS1; fs_exp = [c if c != 'cap' else 'cap_exp' for c in fs]
    for m in MESES:
        tr = G5[(G5['ym'] < m) & G5[y].notna()]; te = G5[(G5['ym'] == m) & G5[y].notna() & G5[y + '_l24'].notna()]
        pred = {'P': te[y + '_l24'].values, 'L_capexp': np.clip(fit_lgb(tr, fs_exp, y).predict(te[fs_exp]), 0, 1)}
        res += metrics(te, y, pred, 'f (cap expansível)', m)
        log(m)
    R = pd.DataFrame(res); R.to_csv(os.path.join(OUT, 'previsao_capexp_mensal.csv'), index=False)
    S = resumo(R); S.to_csv(os.path.join(OUT, 'previsao_capexp_resumo.csv'), index=False); print(S.round(3).to_string(), flush=True)

# ---------------------------------------------------------------- 6. importância por permutação
if '6' in SEC:
    log('6. importância por permutação (modelo final, teste set/2026)')
    rows = []; m = MESES[-1]; rng = np.random.default_rng(0)
    for y in ['f', 'f_grid']:
        fs = BASE + LAGS(y) + SYS1
        tr = G[(G['ym'] < m) & G[y].notna()]; te = G[(G['ym'] == m) & G[y].notna() & G[y + '_l24'].notna()].copy()
        mdl = fit_lgb(tr, fs, y); w = te['pot'].values; obs = te[y].values
        base = np.sum(w * np.abs(np.clip(mdl.predict(te[fs]), 0, 1) - obs))
        for c in fs:
            inc = []
            for r in range(3):
                tp = te[fs].copy(); v = rng.permutation(tp[c].values)
                tp[c] = pd.Categorical(v, categories=tp[c].cat.categories) if isinstance(tp[c].dtype, pd.CategoricalDtype) else v
                inc.append(np.sum(w * np.abs(np.clip(mdl.predict(tp), 0, 1) - obs)) / base - 1)
            rows.append(dict(alvo=y, var=c, aumento_erro=np.mean(inc)))
        log(y, 'ok')
    pd.DataFrame(rows).sort_values(['alvo', 'aumento_erro'], ascending=[True, False]).to_csv(os.path.join(OUT, 'previsao_importancia_permutacao.csv'), index=False)
log('fim')
