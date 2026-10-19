import pandas as pd, numpy as np, matplotlib, os
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8})
R='resultados'; F='figuras'
col={'gnr_ENE':'#e6a100','gnr_CNF':'#2a6fb0','gnr_REL':'#b03a2e'}
lab={'gnr_ENE':'Energy surplus (ENE)','gnr_CNF':'Reliability (CNF)','gnr_REL':'External electrical unavailability (REL)'}
M=pd.read_csv(f'{R}/mensal.csv',index_col=0); M=M[M.index<'2026-10']
fig,ax=plt.subplots(figsize=(7,3.2)); b=np.zeros(len(M)); x=np.arange(len(M))
for c in ['gnr_ENE','gnr_CNF','gnr_REL']:
    v=(100*M[c]/M["pot"]).values; ax.bar(x,v,bottom=b,color=col[c],label=lab[c],width=0.8); b+=v
ax.set_xticks(x[::3]); ax.set_xticklabels(M.index[::3],rotation=0); ax.set_ylabel('Curtailed energy (% of potential)')
ax.legend(frameon=False,fontsize=7,loc='upper left'); ax2=ax.twinx(); ax2.plot(x,M['pot']/1e3,'k-',lw=1); ax2.set_ylabel('Potential PV energy (TWh/month)')
plt.tight_layout(); [plt.savefig(f'{F}/Fig1_monthly.{e}',dpi=300) for e in ('png','tif')]; plt.close()
# Fig2: perfil horário + diário ENE vs carga líquida
Hh=pd.read_csv(f'{R}/perfil_horario_pct.csv',index_col=0); Hh=Hh.loc[6:18]
d=pd.read_csv(f'{R}/diario_sistema.csv',index_col=0,parse_dates=True)
fig,axs=plt.subplots(1,2,figsize=(7,3))
for c in ['gnr_ENE','gnr_CNF','gnr_REL']: axs[0].plot(Hh.index,Hh[c],'-o',ms=3,color=col[c],label=lab[c].split(' (')[0])
axs[0].set_xlabel('Hour of day (local)'); axs[0].set_ylabel('Curtailed (% of potential)'); axs[0].legend(frameon=False,fontsize=7); axs[0].set_title('(a)',loc='left')
sc=axs[1].scatter(d['liq_GW'],100*d['f_ene'],c=d['t'],s=6,cmap='viridis')
axs[1].set_xlabel('Midday net load, SIN (GW)'); axs[1].set_ylabel('Energy-surplus curtailment (%)'); axs[1].set_title('(b)',loc='left')
cb=plt.colorbar(sc,ax=axs[1]); cb.set_label('Years since Apr 2024')
plt.tight_layout(); [plt.savefig(f'{F}/Fig2_drivers.{e}',dpi=300) for e in ('png','tif')]; plt.close()
# Fig3: por UF
U=pd.read_csv(f'{R}/por_uf.csv',index_col=0); U=U[U['pot']>500]
fig,ax=plt.subplots(figsize=(6,3)); b=np.zeros(len(U)); x=np.arange(len(U))
for c in ['gnr_ENE','gnr_CNF','gnr_REL']:
    v=(100*U[c]/U["pot"]).values; ax.bar(x,v,bottom=b,color=col[c],label=lab[c]); b+=v
ax.set_xticks(x); ax.set_xticklabels(U.index); ax.set_ylabel('Curtailed (% of potential), Apr 2024–Oct 2026')
for i,(k,r) in enumerate(U.iterrows()): ax.text(i,b[i]+0.5,f"{r['gnr']/1e3:.1f}",ha='center',fontsize=6)
ax.set_ylim(0,36); ax.legend(frameon=False,fontsize=7,ncol=3,loc='upper center'); plt.tight_layout(); [plt.savefig(f'{F}/Fig3_states.{e}',dpi=300) for e in ('png','tif')]; plt.close()
print('ok')
