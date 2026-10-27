"""Plots from recorded results only."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def plots(variants):
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for name in variants:
        frame=pd.DataFrame(json.loads(Path(f'runs/{name}/history.json').read_text()))
        axes[0].plot(frame.epoch,frame.train_loss,label=name)
        axes[1].plot(frame.epoch,frame.validation_psnr,label=name)
    axes[0].set(xlabel='Epoch',ylabel='Training loss'); axes[1].set(xlabel='Epoch',ylabel='Validation PSNR (dB)')
    for ax in axes: ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig('runs/training_comparison.png',dpi=150); plt.close(fig)
    rows=[]
    for name in variants:
        r=json.loads(Path(f'runs/{name}/test/metrics.json').read_text())
        rows.append({'variant':name,**r['overall']})
    pd.DataFrame(rows).to_csv('runs/rice_test_comparison.csv',index=False)
