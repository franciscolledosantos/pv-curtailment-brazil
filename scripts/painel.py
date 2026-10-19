"""Painel horário usina × hora dos cortes (constrained-off) das fotovoltaicas do ONS, abr/2024–out/2026.

Potencial = geração verificada + geração não realizada apurada (GNRa), em MWh.
O corte de cada meia hora é atribuído à razão informada (REL elétrica externa, CNF confiabilidade,
ENE energética) e à origem (LOC local, SIS sistêmica). Agregação horária (média das duas meias horas → MWh).
Variáveis de sistema: carga horária por subsistema (ONS) e geração eólica de referência/verificada
somada por subsistema (dados de constrained-off das eólicas).
"""
import glob, os
import numpy as np, pandas as pd

B = os.path.join(os.path.dirname(__file__), '..', 'dados')
OUT = os.path.join(os.path.dirname(__file__), '..', 'resultados')
os.makedirs(OUT, exist_ok=True)

pv = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(B, 'ons_pv', '*.parquet')))], ignore_index=True)
for c in ['id_subsistema', 'id_estado', 'nom_usina', 'id_ons', 'cod_razaorestricao', 'cod_origemrestricao',
          'id_pontoconexao', 'nom_pontoconexao', 'dsc_restricao']:
    pv[c] = pv[c].astype(object)
pv = pv.drop_duplicates(['id_ons', 'din_instante'])
for c in ['val_geracao', 'val_geracaonaorealizadaapurada', 'val_disponibilidade', 'val_geracaolimitada']:
    pv[c] = pd.to_numeric(pv[c], errors='coerce').astype(float)
pv['gnr'] = pv['val_geracaonaorealizadaapurada'].fillna(0).clip(lower=0)
pv['ger'] = pv['val_geracao'].fillna(0).clip(lower=0)
pv['pot'] = pv['ger'] + pv['gnr']
for r in ['REL', 'CNF', 'ENE']:
    pv['gnr_' + r] = np.where(pv['cod_razaorestricao'] == r, pv['gnr'], 0.0)
pv['gnr_LOC'] = np.where(pv['cod_origemrestricao'] == 'LOC', pv['gnr'], 0.0)
pv['lim'] = pv['val_geracaolimitada'].notna().astype(float)
pv['hora'] = pv['din_instante'].dt.floor('h')
# meia hora em MWmed -> energia (MWh) = valor/2
agg = {c: 'sum' for c in ['ger', 'gnr', 'pot', 'gnr_REL', 'gnr_CNF', 'gnr_ENE', 'gnr_LOC']}
agg.update(lim='max', val_disponibilidade='mean')
H = pv.groupby(['id_ons', 'hora']).agg(agg).reset_index()
for c in ['ger', 'gnr', 'pot', 'gnr_REL', 'gnr_CNF', 'gnr_ENE', 'gnr_LOC']:
    H[c] = H[c] / 2
meta = pv.sort_values('din_instante').groupby('id_ons').agg(
    nom_usina=('nom_usina', 'last'), uf=('id_estado', 'last'), sub=('id_subsistema', 'last'),
    ponto=('id_pontoconexao', 'last'), nom_ponto=('nom_pontoconexao', 'last'),
    cap=('val_disponibilidade', lambda s: s.quantile(0.95)), inicio=('din_instante', 'min'))
H = H.merge(meta[['uf', 'sub', 'ponto', 'cap']], left_on='id_ons', right_index=True, how='left')
meta.to_csv(os.path.join(OUT, 'usinas_pv.csv'))
# descrições (a partir de set/2025)
dsc = pv[pv['gnr'] > 0].groupby(['cod_razaorestricao', 'dsc_restricao'], dropna=False)['gnr'].sum().div(2e3).sort_values(ascending=False)
dsc.to_csv(os.path.join(OUT, 'gnr_por_descricao_GWh.csv'))

# sistema: carga por subsistema (horária) e eólica por subsistema
cg = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(B, 'carga', '*.parquet')))])
cg['id_subsistema'] = cg['id_subsistema'].astype(str)
cg['val_cargaenergiahomwmed'] = pd.to_numeric(cg['val_cargaenergiahomwmed'], errors='coerce')
cg['din_instante'] = pd.to_datetime(cg['din_instante'])
cg = cg.pivot_table(index='din_instante', columns='id_subsistema', values='val_cargaenergiahomwmed').add_prefix('carga_')
cg['carga_SIN'] = cg.sum(axis=1)
eo = pd.concat([pd.read_parquet(f, columns=['id_subsistema', 'din_instante', 'val_geracao', 'val_geracaonaorealizadaapurada'])
                for f in sorted(glob.glob(os.path.join(B, 'ons_eol', '*.parquet')))])
eo['id_subsistema'] = eo['id_subsistema'].astype(str)
for c in ['val_geracao', 'val_geracaonaorealizadaapurada']:
    eo[c] = pd.to_numeric(eo[c], errors='coerce').astype(float)
eo['pot'] = eo['val_geracao'].fillna(0).clip(lower=0) + eo['val_geracaonaorealizadaapurada'].fillna(0).clip(lower=0)
eo['hora'] = eo['din_instante'].dt.floor('h')
eo = eo.groupby(['hora', 'id_subsistema'])['pot'].sum().div(2).unstack().add_prefix('eol_')
eo['eol_SIN'] = eo.sum(axis=1)
pvs = H.groupby('hora')['pot'].sum().rename('pv_SIN')
S = cg.join(eo, how='left').join(pvs, how='left')
S.index.name = 'hora'
S['liq_SIN'] = S['carga_SIN'] - S['eol_SIN'].fillna(0) - S['pv_SIN'].fillna(0)
S.to_parquet(os.path.join(B, 'sistema_horario.parquet'))
H.to_parquet(os.path.join(B, 'painel_pv_horario.parquet'), index=False)
print('painel', H.shape, 'usinas', H['id_ons'].nunique(), 'pontos', H['ponto'].nunique(), 'horas', H['hora'].nunique())
print('sistema', S.shape, S.index.min(), S.index.max())
tot = H[['ger', 'gnr', 'pot', 'gnr_REL', 'gnr_CNF', 'gnr_ENE', 'gnr_LOC']].sum() / 1e3
print((tot).round(0)); print('corte % do potencial', round(100 * tot['gnr'] / tot['pot'], 1))
