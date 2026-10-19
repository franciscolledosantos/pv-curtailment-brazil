"""Guarda previsões D+1 (alvo total e ENE) para análises de robustez: por UF, por hora, por idade da usina e calibração."""
import os, numpy as np, pandas as pd, lightgbm as lgb, warnings
warnings.filterwarnings('ignore')
B='dados'; OUT='resultados'
G=pd.read_parquet(f'{B}/painel_modelo.parquet')
for c in ['id_ons','uf','sub','ponto']: G[c+'_cat']=G[c].astype('category')
G['ym']=G['hora'].dt.to_period('M')
BASE=['h','dow','mes','feriado','doy_s','doy_c','cap','id_ons_cat','uf_cat','sub_cat','ponto_cat','pot_l24','pot_m7']
SYS1=[c for c in G.columns if c.endswith('_l24') and c.split('_l24')[0] in ['carga_SIN','carga_NE','carga_SE','eol_SIN','eol_NE','pv_SIN','liq_SIN','ene_frac_SIN']]
prim=G.groupby('id_ons')['hora'].transform('min')
out=[]
for y in ['f','f_ene']:
    LAGS=[y+'_l24',y+'_l48',y+'_l168',y+'_m7',y+'_d1']+[z for z in ['f_l24','f_grid_l24','f_ene_l24','f_m7','f_grid_m7','f_ene_m7'] if not z.startswith(y+'_')]
    fs=BASE+LAGS+SYS1
    for m in pd.period_range('2025-04','2026-09',freq='M'):
        tr=G[(G['ym']<m)&G[y].notna()]; te=G[(G['ym']==m)&G[y].notna()&G[y+'_l24'].notna()].copy()
        mdl=lgb.LGBMRegressor(objective='cross_entropy',n_estimators=400,learning_rate=0.05,num_leaves=63,min_child_samples=100,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,verbose=-1)
        mdl.fit(tr[fs],tr[y],sample_weight=tr['pot'].clip(lower=0.5))
        te['p_L']=np.clip(mdl.predict(te[fs]),0,1); te['p_P']=te[y+'_l24']; te['obs']=te[y]; te['alvo']=y
        te['meses_hist']=((te['hora']-prim.loc[te.index]).dt.days/30.4)
        out.append(te[['alvo','id_ons','uf','hora','h','pot','obs','p_L','p_P','meses_hist']]); print(y,m,flush=True)
P=pd.concat(out); P.to_parquet(f'{B}/previsoes_detalhe.parquet',index=False)
def sk(g): 
    e1=(g.pot*(g.p_L-g.obs).abs()).sum(); e0=(g.pot*(g.p_P-g.obs).abs()).sum(); return pd.Series({'skill':1-e1/e0,'curt_GWh':(g.pot*g.obs).sum()/1e3})
r1=P.groupby(['alvo','uf']).apply(sk); r2=P.groupby(['alvo','h']).apply(sk)
P['hist']=pd.cut(P.meses_hist,[-1,6,12,100],labels=['<6 meses','6-12','>12'])
r3=P.groupby(['alvo','hist']).apply(sk)
P['dec']=P.groupby('alvo')['p_L'].transform(lambda s: pd.qcut(s.rank(method='first'),10,labels=False))
cal=P.groupby(['alvo','dec']).apply(lambda g: pd.Series({'prev':np.average(g.p_L,weights=g.pot),'obs':np.average(g.obs,weights=g.pot)}))
for n,r in [('uf',r1),('hora',r2),('historico',r3),('calibracao',cal)]: r.to_csv(f'{OUT}/robustez_{n}.csv'); print(r.round(3))
