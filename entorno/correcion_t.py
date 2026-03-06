#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


def linfit(x,y):

    x=np.asarray(x,dtype=float)
    y=np.asarray(y,dtype=float)

    mask=np.isfinite(x)&np.isfinite(y)

    x=x[mask]
    y=y[mask]

    b,a=np.polyfit(x,y,1)

    yhat=a+b*x

    ss_res=np.sum((y-yhat)**2)
    ss_tot=np.sum((y-y.mean())**2)

    r2=1-ss_res/ss_tot if ss_tot!=0 else np.nan

    return a,b,r2


def pearson(x,y):

    x=np.asarray(x,dtype=float)
    y=np.asarray(y,dtype=float)

    mask=np.isfinite(x)&np.isfinite(y)

    x=x[mask]
    y=y[mask]

    x=x-np.mean(x)
    y=y-np.mean(y)

    return np.sum(x*y)/np.sqrt(np.sum(x*x)*np.sum(y*y))


def main():

    parser=argparse.ArgumentParser(
        description="Corrección de tasa por temperatura"
    )

    parser.add_argument("csv",type=Path)

    parser.add_argument("--time-col",default="index")
    parser.add_argument("--rate-col",default="counts_per_min")
    parser.add_argument("--temp-col",default="temp_roll_med")
    parser.add_argument("--pres-col",default="pres_roll_med")

    args=parser.parse_args()

    df=pd.read_csv(args.csv)

    df[args.time_col]=pd.to_datetime(df[args.time_col])

    t=df[args.time_col]

    R=df[args.rate_col].values
    T=df[args.temp_col].values
    P=df[args.pres_col].values

    T0=np.nanmean(T)

    a,b,r2=linfit(T,R)

    alpha=b

    Rcorr=R-alpha*(T-T0)

    df["rate_temp_corrected"]=Rcorr

    rT_before=pearson(T,R)
    rT_after=pearson(T,Rcorr)

    rP_before=pearson(P,R)
    rP_after=pearson(P,Rcorr)

    outdir=args.csv.parent/"temp_corrected"
    outdir.mkdir(exist_ok=True)

    df.to_csv(outdir/"corrected_rates.csv",index=False)

    # ---------- FIGURA PROFESIONAL ----------

    plt.rcParams.update({

        "font.family":"serif",
        "font.size":9,
        "axes.labelsize":9,
        "axes.titlesize":10,
        "legend.fontsize":8,
        "xtick.labelsize":8,
        "ytick.labelsize":8
    })

    fig,ax=plt.subplots(2,1,figsize=(7,5),sharex=True,constrained_layout=True)

    # tasa

    ax[0].plot(t,R,label="Original",linewidth=1)
    ax[0].plot(t,Rcorr,label="Temp corrected",linewidth=1)

    ax[0].set_ylabel("Counts / min")

    ax[0].legend()

    ax[0].grid(True,linestyle="--",alpha=0.5)

    # temperatura

    ax[1].plot(t,T,color="tab:red",linewidth=1)

    ax[1].set_ylabel("Temperature (°C)")
    ax[1].set_xlabel("Time")

    ax[1].grid(True,linestyle="--",alpha=0.5)

    locator=mdates.AutoDateLocator()

    formatter=mdates.ConciseDateFormatter(locator)

    ax[1].xaxis.set_major_locator(locator)

    ax[1].xaxis.set_major_formatter(formatter)

    fig.suptitle("Temperature correction of Muon rate")

    plt.savefig(outdir/"timeseries_corrected.png",dpi=300)

    plt.savefig(outdir/"timeseries_corrected.pdf")

    plt.close()

    print()
    print("Temperatura media:",T0)
    print("Coeficiente dR/dT:",alpha)
    print()

    print("Correlación temperatura antes :",rT_before)
    print("Correlación temperatura después :",rT_after)
    print()

    print("Correlación presión antes :",rP_before)
    print("Correlación presión después :",rP_after)
    print()

    print("Salida:",outdir)


if __name__=="__main__":
    main()
