import re
import sys
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd


def draw_histo(df, col_name, x_start, bin_width, n_bins, title, output_dir, output_filename, 
        log_y = False, describe = False):
    """
    Create a histogram from a pandas dataframe column

    Args:
        df: a Pandas dataframe
        col_name: a string giving the column from the dataframe you'd like to plot
        x_start: the leftmost x-value in your plot
        bin_width: the desired width of your histogram bins
        n_bins: the desired number of histogram bins
        title: a string giving the title to be printed on your plot
        output_dir: the directory to write your output images to
        output_filename: the name you'd like to give the png output image
    """
    my_plt = df[col_name].plot.hist(
            bins=[bin_width * (x + 0.5) + x_start for x in np.arange(0, n_bins)], 
            edgecolor='black', 
            color='#88d498',
            log=log_y)
    my_plt.set_title(title)
    pd.set_option('precision', 3)
    plt.tight_layout(pad=5)
    if(describe): print(df[col_name].describe())
    plt.figtext(0.01,0.01, df[col_name].describe().loc[['mean','std','count']].to_string(),fontsize='x-large')
    my_plt.ticklabel_format(style='sci', axis='x', scilimits=(0,3)) 
    my_plt.get_figure().savefig(os.path.join(output_dir,output_filename))
    my_plt.get_figure().clf()


if __name__ == "__main__":
    performance_csv = 'data/10k_attacks_12Feb/performance_list.csv'
    output_dir = '.'
    with open (performance_csv) as csv_file:
        df = pd.read_csv(csv_file, 
                dtype={
                    'img_num':str,
                    'epsilon':int,
                    'img_size': int, 
                    'true_box_size':int,
                    'truth_iou_no_noise':float,
                    'truth_iou_noise_avg':float,
                    'ssim_avg':float})  
        print(df.head(10))
        df = df[df['epsilon'] >= 0] # get rid of rows from failed attacks  
        df['true_box_size_frac'] = df['true_box_size'] / df['img_size'] 
        df_small = df[df['true_box_size_frac'] <= 0.038] 
        df_big = df[df['true_box_size_frac'] > 0.038]

        eps_title = 'Epsilon Additive Noise Value to Beat HoG'
        eps_filename = 'eps_hist.png'
        size_title = 'Image Size (Width x Height)'
        size_filename = 'img_size_hist.png'
        box_title = 'True Face Box Size (Width x Height)'
        box_filename = 'true_box_size_hist.png'
        iou_title = 'IoU between Found and True Box Prior to Attack'
        iou_filename = 'truth_iou_no_noise.png'
        noise_iou_title = 'IoU between Found and True Box After Noise Attack (Average)'
        noise_iou_filename = 'truth_iou_noise.png'
        ssim_title = 'Structural Similarity (Average)'
        ssim_filename = 'ssim.png'
        true_box_frac_title = 'True Box Size as a Fraction of Total Image Size'
        true_box_frac_filename = 'true_box_size_frac_hist.png'

        draw_histo(df, 'epsilon', -16, 16, 16, eps_title, output_dir, eps_filename)
        draw_histo(df, 'img_size', 400000, 100000, 21, size_title, output_dir, size_filename) 
        draw_histo(df, 'true_box_size', -15000, 15000, 25, box_title, output_dir, box_filename) 
        draw_histo(df, 'truth_iou_no_noise', -0.05, 0.05, 21, iou_title, output_dir, iou_filename) 
        draw_histo(df, 'truth_iou_noise_avg', -0.05, 0.05, 21, noise_iou_title, output_dir, 
                noise_iou_filename)
        draw_histo(df, 'ssim_avg', 0.48, 0.02, 26, ssim_title, output_dir, ssim_filename, 
                log_y = True)
        draw_histo(df, 'true_box_size_frac', -0.02, 0.02, 31, true_box_frac_title, output_dir, 
                true_box_frac_filename, log_y = True, describe = True)

        small_title = ' (Small Face)'
        small_filename = 'smallface_'
        draw_histo(df_small, 'epsilon', -16, 16, 16, eps_title + small_title, 
                output_dir, small_filename + eps_filename)
        draw_histo(df_small, 'img_size', 400000, 100000, 21, size_title + small_title, 
                output_dir, small_filename + size_filename)
        draw_histo(df_small, 'true_box_size', -15000, 15000, 25, box_title + small_title, 
                output_dir, small_filename + box_filename)
        draw_histo(df_small, 'truth_iou_no_noise', -0.05, 0.05, 21, iou_title + small_title, 
                output_dir, small_filename + iou_filename)
        draw_histo(df_small, 'truth_iou_noise_avg', -0.05, 0.05, 21, noise_iou_title + small_title, 
                output_dir, small_filename + noise_iou_filename)
        draw_histo(df_small, 'ssim_avg', 0.48, 0.02, 26, ssim_title + small_title, 
                output_dir, small_filename + ssim_filename, log_y = True)
        draw_histo(df_small, 'true_box_size_frac', -0.02, 0.02, 31, true_box_frac_title + small_title, 
                output_dir, small_filename + true_box_frac_filename, log_y = True, describe = True)

        big_title = ' (Big Face)'
        big_filename = 'bigface_'
        draw_histo(df_big, 'epsilon', -16, 16, 16, eps_title + big_title,
                output_dir, big_filename + eps_filename)
        draw_histo(df_big, 'img_size', 400000, 100000, 21, size_title + big_title,
                output_dir, big_filename + size_filename)
        draw_histo(df_big, 'true_box_size', -15000, 15000, 25, box_title + big_title,
                output_dir, big_filename + box_filename)
        draw_histo(df_big, 'truth_iou_no_noise', -0.05, 0.05, 21, iou_title + big_title,
                output_dir, big_filename + iou_filename)
        draw_histo(df_big, 'truth_iou_noise_avg', -0.05, 0.05, 21, noise_iou_title + big_title,
                output_dir, big_filename + noise_iou_filename)
        draw_histo(df_big, 'ssim_avg', 0.48, 0.02, 26, ssim_title + big_title,
                output_dir, big_filename + ssim_filename, log_y = True)
        draw_histo(df_big, 'true_box_size_frac', -0.02, 0.02, 31, true_box_frac_title + big_title,
                output_dir, big_filename + true_box_frac_filename, log_y = True, describe = True)

