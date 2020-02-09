import re
import sys
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd


def draw_histo(df, col_name, x_start, bin_width, n_bins, title, output_dir, output_filename):
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
    plt = df[col_name].plot.hist(
            bins=[bin_width * (x + 0.5) + x_start for x in np.arange(0, n_bins)], 
            edgecolor='black', 
            color='#88d498')
    plt.set_title(title)
    print(col_name + ' max: ' + str(df[col_name].max()) + ', min: ' + str(df[col_name].min()))
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,3)) 
    plt.get_figure().savefig(os.path.join(output_dir,output_filename))
    plt.get_figure().clf()


if __name__ == "__main__":
    performance_csv = 'data/10k_imgs_addnoise/performance_list.csv'
    output_dir = '/home/ubuntu/hideface'
    with open (performance_csv) as csv_file:
        df = pd.read_csv(csv_file, 
                dtype={
                    'img_num':str,
                    'epsilon':int,
                    'img_size': int, 
                    'true_box_size':int,
                    'truth_iou_no_noise':float,
                    'truth_iou_noise':float,
                    'ssim':float})  
        print(df.head(10))
        df = df[df['epsilon'] >= 0] # get rid of rows from failed attacks  
    
        eps_title = 'Epsilon Additive Noise Value to Beat HoG'
        eps_filename = 'eps_hist.png'
        draw_histo(df, 'epsilon', -16, 16, 16, eps_title, output_dir, eps_filename)
        size_title = 'Image Size (Width x Height)'
        size_filename = 'img_size_hist.png'
        draw_histo(df, 'img_size', 400000, 100000, 21, size_title, output_dir, size_filename) 
        box_title = 'True Face Box Size (Width x Height)'
        box_filename = 'true_box_size_hist.png'
        draw_histo(df, 'true_box_size', -15000, 15000, 25, box_title, output_dir, box_filename) 
        iou_title = 'IoU between Found and True Box Prior to Attack'
        iou_filename = 'truth_iou_no_noise.png'
        draw_histo(df, 'truth_iou_no_noise', -0.05, 0.05, 21, iou_title, output_dir, iou_filename) 
        iou_title = 'IoU between Found and True Box After Noise Attack'
        iou_filename = 'truth_iou_noise.png'
        draw_histo(df, 'truth_iou_noise', -0.05, 0.05, 21, iou_title, output_dir, iou_filename)
        draw_histo(df, 'ssim', -0.05, 0.05, 21, 'Structural Similarity', output_dir, 'ssim.png')

        bin_width = 0.01
        x_start = -0.01
        n_bins = 31
        true_box_size_frac_plt = (df['true_box_size']/df['img_size']).plot.hist(
                bins=[bin_width * (x + 0.5) + x_start for x in np.arange(0, n_bins)],
                edgecolor='black', 
                color='#88d498')
        print('true_box_size_frac max: ' 
            + str((df['true_box_size']/df['img_size']).max()) 
            + ', min: ' + str((df['true_box_size']/df['img_size']).min()))
        true_box_size_frac_plt.set_title('True Box Size as a Fraction of Total Image Size')
        true_box_size_frac_plt.get_figure().savefig(output_dir + 'true_box_size_frac_hist.png')
        true_box_size_frac_plt.get_figure().clf()

