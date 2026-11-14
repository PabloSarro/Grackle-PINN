"""Compares the A3 driver with the old one (binary built from tag report-june2026) on the 8 report-figure ICs.
Run from the repository root; needs results/old_driver. Prints max |d ln| of T, nHI, nHII at the old output times."""
import numpy as np, pandas as pd, subprocess, glob, os, sys
D='scripts/datagen/cooling-full-integration-100yr'; O='results/old_driver'
for f in sorted(glob.glob('data/Plot_Files/*_Fam_*.dat')):
    d=pd.read_csv(f,comment='#',sep=r'\s+',header=None)
    T0=d[3][0]; nH=d[6][0]+d[7][0]; x0=d[7][0]/nH
    n=os.path.basename(f)[:-4]
    ok=[]
    for tag,exe in (('old',O),('new',D)):
        try:
            subprocess.run([exe,"%.17g"%T0,"%.17g"%nH,"%.17g"%x0,f'results/ver/{n}_{tag}.dat'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=True,timeout=60); ok.append(True)
        except Exception as e:
            print('  ',tag,repr(e)[:200]); ok.append(False)
    print(n,f"T0={T0:.6g} nH={nH:.6g} x0={x0:.6g}","old ok" if ok[0] else "old FAIL","new ok" if ok[1] else "new FAIL",flush=True)
    if not all(ok): continue
    a=pd.read_csv(f'results/ver/{n}_old.dat',comment='#',sep=r'\s+',header=None)
    b=pd.read_csv(f'results/ver/{n}_new.dat',comment='#',sep=r'\s+',header=None)
    res=[np.abs(np.interp(a[1].values,b[1].values,np.log(b[c].values))-np.log(a[c].values)).max() for c in (3,6,7)]
    print("   max |dln| T, nHI, nHII at old times:",*(f"{r:.2e}" for r in res),"| rows",len(a),len(b),"| new dt",np.diff(b[1]).min(),np.diff(b[1]).max(),"| old dt",round(np.diff(a[1]).min(),1),round(np.diff(a[1]).max(),1),flush=True)
