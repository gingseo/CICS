from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from .common import clean,write_json

REGIONS = ['BG','underflow','VT','T','S','M','L']


def describe(values):
    a = np.asarray(values,dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return dict(count=0,mean=None,std=None,median=None,q1=None,q3=None)
    return dict(count=len(a),mean=float(a.mean()),std=float(a.std()),median=float(np.median(a)),
                q1=float(np.quantile(a,.25)),q3=float(np.quantile(a,.75)))


def summarize(objects,background):
    rows=[]
    groups=[]
    for region in REGIONS:
        values = background.get('score',pd.Series(dtype=float)) if region == 'BG' else objects.loc[objects['size_bin']==region,'score']
        d = describe(values)
        groups.append(np.asarray(pd.to_numeric(values).dropna(),dtype=float))
        rows.append(dict(region=region,sample_unit='image_bg_mean' if region=='BG' else 'object',
                         excluded_count=int(pd.isna(values).sum()),
                         n_images=int(background.image_id.nunique()) if region=='BG' and 'image_id' in background else (int(objects.loc[objects.size_bin==region,'image_id'].nunique()) if region!='BG' else len(background)),
                         n_objects=0 if region=='BG' else int((objects.size_bin==region).sum()),
                         n_valid_locations=int(background.get('valid_sample_count',pd.Series(dtype=float)).sum()) if region=='BG' else int(objects.loc[objects.size_bin==region].get('valid_sample_count',pd.Series(dtype=float)).sum()),**d))
    n = int(background.get('valid_sample_count',pd.Series(dtype=float)).sum())
    total = float(background.get('sum',pd.Series(dtype=float)).sum())
    bgmean = total/n if n else None
    for row in rows:
        m=row['mean']
        row['difference_from_global_bg'] = m-bgmean if m is not None and bgmean is not None else None
        row['relative_difference_from_global_bg'] = (m-bgmean)/bgmean if m is not None and bgmean is not None and abs(bgmean)>1e-8 else None
    summary = pd.DataFrame(rows)
    result = dict(global_bg=dict(mean=bgmean,n_valid_locations=n,n_images=len(background),sample_unit='input_pixel'),
                  n_objects=len(objects),n_images=int(objects['image_id'].nunique()) if len(objects) else 0,
                  std_definition='population std; not confidence interval',
                  statistical_caveat='Objects cluster within images; p-values are auxiliary, not independent-image evidence.',
                  correlations={},pairwise=[],multiple_comparisons='Holm across bin-vs-BG Mann-Whitney tests')
    for col in ('original_size','input_size'):
        finite=objects[[col,'score']].apply(pd.to_numeric).dropna()
        for method,fn in [('spearman',stats.spearmanr),('pearson',stats.pearsonr)]:
            if len(finite)>2 and finite[col].nunique()>1 and finite.score.nunique()>1:
                corr,p = fn(finite[col],finite.score)
                result['correlations'][col+'_'+method] = dict(coefficient=corr,pvalue=p,n=len(finite))
            else:
                result['correlations'][col+'_'+method] = None
    nonempty=[g for g in groups if len(g)]
    try:
        k=stats.kruskal(*nonempty) if len(nonempty)>1 and np.unique(np.concatenate(nonempty)).size>1 else None
        result['kruskal_wallis'] = dict(statistic=k.statistic,pvalue=k.pvalue) if k else None
    except ValueError:
        result['kruskal_wallis']=None
    for name,g in zip(REGIONS[1:],groups[1:]):
        if len(groups[0]) and len(g):
            u=stats.mannwhitneyu(g,groups[0],alternative='two-sided')
            result['pairwise'].append(dict(region=name,statistic=u.statistic,pvalue=u.pvalue))
    ranked=sorted(result['pairwise'],key=lambda x:x['pvalue'])
    prev=0.
    for i,r in enumerate(ranked):
        prev=max(prev,min(1.,r['pvalue']*(len(ranked)-i)))
        r['pvalue_holm']=prev
    return summary,clean(result)


def figures(objects,background,summary,out,seed=128):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    def save(fig,name):
        fig.tight_layout()
        for ext in ('png','pdf','svg'):
            fig.savefig(out/f'{name}.{ext}',dpi=150)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    ax.bar(summary.region,summary['mean'].astype(float),yerr=summary['std'].astype(float).fillna(0),capsize=3)
    ax.set(ylabel='Raw discrepancy',title='Mean ± population std; BG = image means')
    save(fig,'bin_means')
    fig,ax=plt.subplots(figsize=(8,4))
    groups=[];labels=[]
    for region in REGIONS:
        g=background.score.dropna() if region=='BG' else objects.loc[objects.size_bin==region,'score'].dropna()
        if len(g):
            groups.append(g);labels.append(region)
    if groups:
        ax.boxplot(groups,tick_labels=labels,whis=1.5,showfliers=False)
    ax.set(ylabel='Raw discrepancy',title='Quartiles, 1.5 IQR whiskers; BG=image, others=object')
    save(fig,'boxplot')
    finite=objects.dropna(subset=['score'])
    sampled=finite.sample(min(20000,len(finite)),random_state=seed)
    for col in ('original_size','input_size'):
        fig,ax=plt.subplots(figsize=(7,4))
        ax.scatter(sampled[col],sampled.score,s=3,alpha=.25)
        if len(sampled): ax.set_xscale('log')
        ax.set(xlabel=f'{col}: sqrt(w*h), pixels (log scale)',ylabel='Raw discrepancy')
        save(fig,'scatter_'+col)
    write_json(out/'display_manifest.json',dict(scatter_seed=seed,scatter_count=len(sampled),
               full_object_count=len(objects),box_whisker='1.5 IQR',fliers_displayed=False))


def export_tables(objects,background,out,seed=128):
    out=Path(out)
    objects.to_csv(out/'objects.csv',index=False)
    background.to_csv(out/'background.csv',index=False)
    summary,statistics=summarize(objects,background)
    summary.to_csv(out/'summary.csv',index=False)
    write_json(out/'summary.json',summary.to_dict('records'))
    write_json(out/'statistics.json',statistics)
    figures(objects,background,summary,out/'figures',seed)
