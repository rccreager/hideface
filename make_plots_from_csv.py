import re
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd

performance_csv = 'data/51_addnoise/performance_list.csv'
with open (performance_csv) as csv_file:
    df = pd.read_csv(csv_file, dtype={'img_num':str,'epsilon':int,'img_size': int, 'true_box_size':int,'truth_iou_no_noise':float,'truth_iou_noise':float})  
    print(df.head(10))
   
    ######################################

    df_attacks = df[df['epsilon'] >= 0] # get rid of rows from failed attacks  

    ######################################

    df_smalleps = df_attacks[df_attacks['epsilon'] < 64]
    df_medeps = df_attacks[df_attacks['epsilon'] >= 64]
    df_medeps = df_medeps[df_medeps['epsilon'] < 112]
    df_largeeps = df_attacks[df_attacks['epsilon'] >= 112]

    bin_width = 100000
    x_start = 400000
    n_bins = 21
    img_size_smalleps_plt = df_smalleps['img_size'].plot.hist(bins=[bin_width*(x+0.5)+x_start for x in np.arange(0,n_bins)], edgecolor='black', color='#88d498')
    img_size_smalleps_plt.set_title('Image Size (Width x Height) for Images with Eps < 64')
    print('img_size_smalleps max: ' + str(df_smalleps['img_size'].max()) + ', min: ' + str(df_smalleps['img_size'].min()))
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    img_size_smalleps_plt.get_figure().savefig('img_size_smalleps_hist.png')
    img_size_smalleps_plt.get_figure().clf()

    bin_width = 100000
    x_start = 400000
    n_bins = 21
    img_size_medeps_plt = df_medeps['img_size'].plot.hist(bins=[bin_width*(x+0.5)+x_start for x in np.arange(0,n_bins)], edgecolor='black', color='#88d498')
    img_size_medeps_plt.set_title('Image Size (Width x Height) for Images with [64,112) Eps')
    print('img_size_medeps max: ' + str(df_medeps['img_size'].max()) + ', min: ' + str(df_medeps['img_size'].min()))
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    img_size_medeps_plt.get_figure().savefig('img_size_medeps_hist.png')
    img_size_medeps_plt.get_figure().clf()

    bin_width = 100000
    x_start = 400000
    n_bins = 21
    img_size_largeeps_plt = df_largeeps['img_size'].plot.hist(bins=[bin_width*(x+0.5)+x_start for x in np.arange(0,n_bins)], edgecolor='black', color='#88d498')
    img_size_largeeps_plt.set_title('Image Size (Width x Height) for Images with Eps >= 112')
    print('img_size_largeeps max: ' + str(df_largeeps['img_size'].max()) + ', min: ' + str(df_largeeps['img_size'].min()))
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    img_size_largeeps_plt.get_figure().savefig('img_size_largeeps_hist.png')
    img_size_largeeps_plt.get_figure().clf()

    bin_width = 15000
    x_start = -15000
    n_bins = 25
    true_box_size_smalleps_plt = df_smalleps['true_box_size'].plot.hist(bins=[bin_width*(x+0.5)+x_start for x in np.arange(0,n_bins)], edgecolor='black', color='#88d498')
    print('true_box_size_smalleps max: ' + str(df_smalleps['true_box_size'].max()) + ', min: ' + str(df_smalleps['true_box_size'].min()))
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    true_box_size_smalleps_plt.set_title('True Face Box Size (Width x Height) for Images with Eps < 64')
    true_box_size_smalleps_plt.get_figure().savefig('true_box_size_smalleps_hist.png')
    true_box_size_smalleps_plt.get_figure().clf()

    bin_width = 15000
    x_start = -15000
    n_bins = 25
    true_box_size_medeps_plt = df_medeps['true_box_size'].plot.hist(bins=[bin_width*(x+0.5)+x_start for x in np.arange(0,n_bins)], edgecolor='black', color='#88d498')
    print('true_box_size_medeps max: ' + str(df_medeps['true_box_size'].max()) + ', min: ' + str(df_medeps['true_box_size'].min()))
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    true_box_size_medeps_plt.set_title('True Face Box Size (Width x Height) for Images with [64,112) Eps')
    true_box_size_medeps_plt.get_figure().savefig('true_box_size_medeps_hist.png')
    true_box_size_medeps_plt.get_figure().clf()

    bin_width = 15000
    x_start = -15000
    n_bins = 25
    true_box_size_largeeps_plt = df_largeeps['true_box_size'].plot.hist(bins=[bin_width*(x+0.5)+x_start for x in np.arange(0,n_bins)], edgecolor='black', color='#88d498')
    print('true_box_size_largeeps max: ' + str(df_largeeps['true_box_size'].max()) + ', min: ' + str(df_largeeps['true_box_size'].min()))
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    true_box_size_largeeps_plt.set_title('True Face Box Size (Width x Height) for Images with Eps >= 112')
    true_box_size_largeeps_plt.get_figure().savefig('true_box_size_largeeps_hist.png')
    true_box_size_largeeps_plt.get_figure().clf()

    bin_width = 0.01
    x_start = -0.01
    n_bins = 31
    true_box_size_frac_smalleps_plt = (df_smalleps['true_box_size']/df_smalleps['img_size']).plot.hist(bins=[bin_width*(x+0.5)+x_start for x in np.arange(0,n_bins)], edgecolor='black', color='#88d498')
    print('true_box_size_frac_smalleps max: ' + str((df_smalleps['true_box_size']/df_smalleps['img_size']).max()) + ', min: ' + str((df_smalleps['true_box_size']/df_smalleps['img_size']).min()))
    true_box_size_frac_smalleps_plt.set_title('True Box Size as a Fraction of Total Image Size for Images with Eps < 64')
    true_box_size_frac_smalleps_plt.get_figure().savefig('true_box_size_frac_smalleps_hist.png')
    true_box_size_frac_smalleps_plt.get_figure().clf() 

    bin_width = 0.01
    x_start = -0.01
    n_bins = 31
    true_box_size_frac_medeps_plt = (df_medeps['true_box_size']/df_medeps['img_size']).plot.hist(bins=[bin_width*(x+0.5)+x_start for x in np.arange(0,n_bins)], edgecolor='black', color='#88d498')
    print('true_box_size_frac_medeps max: ' + str((df_medeps['true_box_size']/df_medeps['img_size']).max()) + ', min: ' + str((df_medeps['true_box_size']/df_medeps['img_size']).min()))
    true_box_size_frac_medeps_plt.set_title('True Box Size as a Fraction of Total Image Size for Images with [64,112) Eps')
    true_box_size_frac_medeps_plt.get_figure().savefig('true_box_size_frac_medeps_hist.png')
    true_box_size_frac_medeps_plt.get_figure().clf()

    bin_width = 0.01
    x_start = -0.01
    n_bins = 31
    true_box_size_frac_largeeps_plt = (df_largeeps['true_box_size']/df_largeeps['img_size']).plot.hist(bins=[bin_width*(x+0.5)+x_start for x in np.arange(0,n_bins)], edgecolor='black', color='#88d498')
    print('true_box_size_frac_largeeps max: ' + str((df_largeeps['true_box_size']/df_largeeps['img_size']).max()) + ', min: ' + str((df_largeeps['true_box_size']/df_largeeps['img_size']).min()))
    true_box_size_frac_largeeps_plt.set_title('True Box Size as a Fraction of Total Image Size for Images with Eps >= 112')
    true_box_size_frac_largeeps_plt.get_figure().savefig('true_box_size_frac_largeeps_hist.png')
    true_box_size_frac_largeeps_plt.get_figure().clf()

    ######################################

    bin_width = 16
    x_start = -16
    n_bins = 16
    df_attacks_smallimg = df_attacks[df_attacks['img_size'] < 1500000]
    eps_smallimg_plt = df_attacks_smallimg['epsilon'].plot.hist(bins=[bin_width*(x+0.5)+x_start for x in np.arange(0,n_bins)], edgecolor='black', color='#88d498')
    eps_smallimg_plt.set_title('Epsilon Additive Noise Value to Beat HoG for Images <1.5e6 sqpix')
    print('epsilon_smallimg max: ' + str(df_attacks_smallimg['epsilon'].max()) + ', min: ' + str(df_attacks_smallimg['epsilon'].min()))
    eps_smallimg_plt.xaxis.set_major_locator(MultipleLocator(16))
    eps_smallimg_plt.get_figure().savefig('eps_smallimg_hist.png')
    eps_smallimg_plt.get_figure().clf()

    bin_width = 16
    x_start = -16
    n_bins = 16
    df_attacks_medimg = df_attacks[df_attacks['img_size'] >= 1500000]
    df_attacks_medimg = df_attacks_medimg[df_attacks_medimg['img_size'] < 1750000]
    eps_medimg_plt = df_attacks_medimg['epsilon'].plot.hist(bins=[bin_width*(x+0.5)+x_start for x in np.arange(0,n_bins)], edgecolor='black', color='#88d498')
    eps_medimg_plt.set_title('Epsilon Additive Noise Value to Beat HoG for Images [1.5,1.75)e6 sqpix')
    print('epsilon_medimg max: ' + str(df_attacks_medimg['epsilon'].max()) + ', min: ' + str(df_attacks_medimg['epsilon'].min()))
    eps_medimg_plt.xaxis.set_major_locator(MultipleLocator(16))
    eps_medimg_plt.get_figure().savefig('eps_medimg_hist.png')
    eps_medimg_plt.get_figure().clf()

    bin_width = 16
    x_start = -16
    n_bins = 16
    df_attacks_largeimg = df_attacks[df_attacks['img_size'] >= 1750000]
    eps_largeimg_plt = df_attacks_largeimg['epsilon'].plot.hist(bins=[bin_width*(x+0.5)+x_start for x in np.arange(0,n_bins)], edgecolor='black', color='#88d498')
    eps_largeimg_plt.set_title('Epsilon Additive Noise Value to Beat HoG for Images >=1.75e6 sqpix')
    print('epsilon_largeimg max: ' + str(df_attacks_largeimg['epsilon'].max()) + ', min: ' + str(df_attacks_largeimg['epsilon'].min()))
    eps_largeimg_plt.xaxis.set_major_locator(MultipleLocator(16))
    eps_largeimg_plt.get_figure().savefig('eps_largeimg_hist.png')
    eps_largeimg_plt.get_figure().clf()

    ######################################

    bin_width = 16
    x_start = -16
    n_bins = 16
    eps_plt = df_attacks['epsilon'].plot.hist(bins=[bin_width*(x+0.5)+x_start for x in np.arange(0,n_bins)], edgecolor='black', color='#88d498')
    eps_plt.set_title('Epsilon Additive Noise Value to Beat HoG')
    print('epsilon max: ' + str(df_attacks['epsilon'].max()) + ', min: ' + str(df_attacks['epsilon'].min()))
    eps_plt.xaxis.set_major_locator(MultipleLocator(16))
    eps_plt.get_figure().savefig('eps_hist.png')
    eps_plt.get_figure().clf()

    bin_width = 100000
    x_start = 400000
    n_bins = 21
    img_size_plt = df_attacks['img_size'].plot.hist(bins=[bin_width*(x+0.5)+x_start for x in np.arange(0,n_bins)], edgecolor='black', color='#88d498')
    img_size_plt.set_title('Image Size (Width x Height)')
    print('img_size max: ' + str(df_attacks['img_size'].max()) + ', min: ' + str(df_attacks['img_size'].min()))    
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    img_size_plt.get_figure().savefig('img_size_hist.png')
    img_size_plt.get_figure().clf()

    bin_width = 15000
    x_start = -15000
    n_bins = 25
    true_box_size_plt = df_attacks['true_box_size'].plot.hist(bins=[bin_width*(x+0.5)+x_start for x in np.arange(0,n_bins)], edgecolor='black', color='#88d498')
    print('true_box_size max: ' + str(df_attacks['true_box_size'].max()) + ', min: ' + str(df_attacks['true_box_size'].min()))
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    true_box_size_plt.set_title('True Face Box Size (Width x Height)')
    true_box_size_plt.get_figure().savefig('true_box_size_hist.png')
    true_box_size_plt.get_figure().clf()

    bin_width = 0.01
    x_start = -0.01
    n_bins = 31
    true_box_size_frac_plt = (df_attacks['true_box_size']/df_attacks['img_size']).plot.hist(bins=[bin_width*(x+0.5)+x_start for x in np.arange(0,n_bins)], edgecolor='black', color='#88d498')
    print('true_box_size_frac max: ' + str((df_attacks['true_box_size']/df_attacks['img_size']).max()) + ', min: ' + str((df_attacks['true_box_size']/df_attacks['img_size']).min()))
    true_box_size_frac_plt.set_title('True Box Size as a Fraction of Total Image Size')
    true_box_size_frac_plt.get_figure().savefig('true_box_size_frac_hist.png')
    true_box_size_frac_plt.get_figure().clf()

    bin_width = 0.05
    x_start = -0.05
    n_bins = 21
    truth_iou_no_noise_plt = df_attacks['truth_iou_no_noise'].plot.hist(bins=[bin_width*(x+0.5)+x_start for x in np.arange(0,n_bins)], edgecolor='black', color='#88d498')
    print('truth_iou_no_noise max: ' + str(df_attacks['truth_iou_no_noise'].max()) + ', min: ' + str(df_attacks['truth_iou_no_noise'].min())) 
    truth_iou_no_noise_plt.set_title('IoU between Found and True Box Prior to Attack')
    truth_iou_no_noise_plt.get_figure().savefig('truth_iou_no_noise.png')
    truth_iou_no_noise_plt.get_figure().clf()

    bin_width = 0.05
    x_start = -0.05
    n_bins = 21
    truth_iou_noise_plt = df_attacks['truth_iou_noise'].plot.hist(bins=[bin_width*(x+0.5)+x_start for x in np.arange(0,n_bins)], edgecolor='black', color='#88d498')
    print('truth_iou_noise max: ' + str(df_attacks['truth_iou_noise'].max()) + ', min: ' + str(df_attacks['truth_iou_noise'].min())) 
    truth_iou_noise_plt.set_title('IoU between Found and True Box After Noise Attack')
    truth_iou_noise_plt.get_figure().savefig('truth_iou_noise.png')
    truth_iou_noise_plt.get_figure().clf()


