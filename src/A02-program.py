#!/bin/python
# Author: Hugh Pearse
# Installation instructions: pip install numpy pandas matplotlib xlrd ipywidgets
import numpy as np
import pandas as pd
import matplotlib  
matplotlib.use('TkAgg')   
matplotlib.rcParams['toolbar'] = 'None'
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.pyplot import figure
from matplotlib import cm as cm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backend_bases import key_press_handler
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1 import make_axes_locatable
import Tkinter as tk
import ttk as ttk
from scipy import stats
from scipy.stats import norm, kstest, iqr

# Read raw data files
df_1 = pd.read_csv("./../data/old-new-borough-codes.csv")
df_2 = pd.read_excel("./../data/childhood-obesity-borough.xls", skiprows=[0,1,2,3], header=None, sheet_name="2011-12") # https://files.datapress.com/london/dataset/prevalence-childhood-obesity-borough/2017-12-11T13:34:04.63/childhood-obesity-borough.xls
df_3 = pd.read_excel("./../data/children-in-poverty.xls", skiprows=[0,1,2], header=None, sheet_name="2011") # https://files.datapress.com/london/dataset/children-poverty-borough/2017-01-26T16:39:49/children-in-poverty.xls
df_4 = pd.read_csv("./../data/crime rates.csv") # https://files.datapress.com/london/dataset/recorded_crime_rates/2018-03-05T11:54:29.74/crime%20rates.csv
df_5 = pd.read_excel("./../data/ratio-house-price-earnings-residence-based.xls", sheet_name="Median Earnings to Prices ratio") # https://data.london.gov.uk/download/ratio-house-prices-earnings-borough/122ea18a-cb44-466e-a314-e0c62a32529e/ratio-house-price-earnings-residence-based.xls
df_6 = pd.read_excel("./../data/statutory-homelessness-borough.xls", skiprows=[0,1,2,3,4,5,6,7,8,9,10], header=None, sheet_name="2011-12") # https://data.london.gov.uk/download/homelessness/7b28ad56-5698-4ea0-9a94-f301fcb62b3e/statutory-homelessness-borough.xls
df_7 = pd.read_csv("./../data/modelled-household-income-estimates-borough.csv") # https://files.datapress.com/london/dataset/household-income-estimates-small-areas/2018-02-16T16:59:58.52/modelled-household-income-estimates-borough.csv
df_8 = pd.read_excel("./../data/immunisation-rates-1st-birthday-pct.xls", skiprows=[0], sheet_name="2011-12") # https://data.london.gov.uk/download/immunisation-rates-children-1st-2nd-and-5th-birthdays/db8970ad-9152-4535-b9e6-26df1bafcd85/immunisation-rates-1st-birthday-pct.xls
df_8.columns = df_8.columns.str.strip()
df_9 = pd.read_csv("./../data/land-registry-house-prices-borough.csv") # https://data.london.gov.uk/download/average-house-prices/b1b0079e-698c-4c0b-b8c7-aa6189590ca4/land-registry-house-prices-borough.csv
df_10 = pd.read_excel("./../data/life-expectancy-ward.xls", skiprows=[0], sheet_name="Life expectancy at 65 - persons") # https://files.datapress.com/london/dataset/life-expectancy-birth-and-age-65-ward/2016-02-09T14:23:39/life-expectancy-ward.xls
df_11 = pd.read_csv("./../data/Qualifications-of-working-age-NVQ.csv") # https://files.datapress.com/london/dataset/qualifications-working-age-population-nvq-borough/2018-02-27T15:10:23.38/Qualifications-of-working-age-NVQ.csv
df_12 = pd.read_csv("./../data/Survival rates.csv") # https://files.datapress.com/london/dataset/business-demographics-and-survival-rates-borough/2018-03-05T11:39:32.74/Survival%20rates.csv
df_13 = pd.read_excel("./../data/workless-households-borough.xls", skiprows=[0,1,2,3], header=None, sheet_name="2011") # https://data.london.gov.uk/download/workless-households-borough/e38fa13b-eea2-425f-a97f-52f8f722e73d/workless-households-borough.xls
df_14 = pd.read_csv("./../data/gcse-results.csv") # https://files.datapress.com/london/dataset/gcse-results-by-borough/2018-02-26T15:39:56.40/gcse-results.csv

# Rename columns to more appropriate names for display
df_2.rename(columns={df_2.columns[0]: "new_borough_code"}, inplace=True)
df_2.rename(columns={df_2.columns[2]: "prevalence of underweight ages 4 and 5"}, inplace=True) # 
df_2.rename(columns={df_2.columns[8]: "prevalence of healthy weight children age 10 and 11"}, inplace=True) # 
df_2.rename(columns={df_2.columns[16]: "prevalence of obese children age 10 and 11"}, inplace=True) # 
df_3.rename(columns={df_3.columns[0]: "new_borough_code"}, inplace=True)
df_3.rename(columns={df_3.columns[9]: "prevelance children under 16 in low-income families"}, inplace=True) # 
df_4.rename(columns={"Code": "new_borough_code"}, inplace=True)
df_4.rename(columns={"Rate": "crime rate"}, inplace=True) # 
df_5.rename(columns={"New Code": "new_borough_code"}, inplace=True)
df_5.rename(columns={df_5.columns[12]: "house price to earnings ratio"}, inplace=True) # 
df_6.rename(columns={df_6.columns[1]: "new_borough_code"}, inplace=True)
df_6.rename(columns={df_6.columns[12]: "rate of homelessness per 1000 households"}, inplace=True) # 
df_7.rename(columns={"Code": "new_borough_code"}, inplace=True)
df_7.rename(columns={"Income": "median household income"}, inplace=True) #
df_8.rename(columns={"Code": "primary_care_trust_code"}, inplace=True)
df_8.rename(columns={df_8.columns[3]: "vaccination rate for DTaP IPV Hib (1 year old)"}, inplace=True) # 
df_8.rename(columns={"Pneumococcal Disease (PCV)": "vaccination rate for PCV (1 year old)"}, inplace=True) # 
df_8.rename(columns={"MenC": "vaccination rate for MenC (1 year old)"}, inplace=True) #
df_9.rename(columns={"Code": "new_borough_code"}, inplace=True)
df_9.rename(columns={"Value": "median house value"}, inplace=True) # 
df_10.rename(columns={"New Code": "new_borough_code"}, inplace=True)
df_10.rename(columns={df_10.columns[28]: "life expectancy at age 65"}, inplace=True) # 
df_11.rename(columns={"Code": "new_borough_code"}, inplace=True)
df_11.rename(columns={df_11.columns[3]: "qual_type"}, inplace=True)
df_11.rename(columns={"percent": "university degree prevalence"}, inplace=True) # 
df_12.rename(columns={"Code": "new_borough_code"}, inplace=True)
df_12.rename(columns={"5_year_survival_%": "rate of 5 year business survival"}, inplace=True) # 
df_13.rename(columns={df_13.columns[0]: "old_borough_code"}, inplace=True)
df_13.rename(columns={df_13.columns[3]: "prevalence of households with full employment"}, inplace=True) # 
df_13.rename(columns={df_13.columns[5]: "prevalence of households with partial employment"}, inplace=True) # 
df_13.rename(columns={df_13.columns[7]: "prevalence of households with no employment"}, inplace=True) # 
df_14.rename(columns={"Code": "new_borough_code"}, inplace=True)
df_14.rename(columns={"Attainment8": "average GCSE score"}, inplace=True)

# Set indexes on appropriate columns
df_1.set_index("new_borough_code")
df_2.set_index("new_borough_code")
df_3.set_index("new_borough_code")
df_4.set_index("new_borough_code")
df_5.set_index("new_borough_code")
df_6.set_index("new_borough_code")
df_7.set_index("new_borough_code")
df_8.set_index("primary_care_trust_code")
df_9.set_index("new_borough_code")
df_10.set_index("new_borough_code")
df_11.set_index("new_borough_code")
df_12.set_index("new_borough_code")
df_13.set_index("old_borough_code")
df_14.set_index("new_borough_code")

# Join all DataFrames to a single DataFrame(df_1) using various indexes
df_1 = pd.merge(df_1, df_2[["new_borough_code","prevalence of obese children age 10 and 11"]], on='new_borough_code', how="left")
df_1 = pd.merge(df_1, df_2[["new_borough_code","prevalence of underweight ages 4 and 5"]], on='new_borough_code', how="left")
df_1 = pd.merge(df_1, df_2[["new_borough_code","prevalence of healthy weight children age 10 and 11"]], on='new_borough_code', how="left")
df_1 = pd.merge(df_1, df_3[["new_borough_code","prevelance children under 16 in low-income families"]], on='new_borough_code', how="left")
df_1 = pd.merge(df_1, df_4.loc[(df_4["Year"] == "2011-12") & (df_4["Offences"] == "All recorded offences"), ["new_borough_code", "crime rate"]], on='new_borough_code', how="left")
df_1 = pd.merge(df_1, df_5[["new_borough_code","house price to earnings ratio"]], on='new_borough_code', how="left")
df_1 = pd.merge(df_1, df_6[["new_borough_code","rate of homelessness per 1000 households"]], on='new_borough_code', how="left")
df_1 = pd.merge(df_1, df_7.loc[(df_7["Year"] == "2011/12") & (df_7["Measure"] == "Median"), ["new_borough_code", "median household income"]], on='new_borough_code', how="left")
df_1.set_index("primary_care_trust_code")
df_1 = pd.merge(df_1, df_8[["primary_care_trust_code","vaccination rate for DTaP IPV Hib (1 year old)", "vaccination rate for MenC (1 year old)", "vaccination rate for PCV (1 year old)"]], on='primary_care_trust_code', how="left")
df_1.set_index("new_borough_code")
df_1 = pd.merge(df_1, df_9.loc[(df_9["Year"] == "Year ending Jun 2011") & (df_9["Measure"] == "Median"), ["new_borough_code", "median house value"]], on='new_borough_code', how="left")
df_1 = pd.merge(df_1, df_10.loc[(df_10["Geography"] == "Local Authority"), ["new_borough_code", "life expectancy at age 65"]], on='new_borough_code', how="left")
df_1 = pd.merge(df_1, df_11.loc[(df_11["Year"] == 2011) & (df_11["qual_type"] == "NVQ4+"), ["new_borough_code", "university degree prevalence"]], on='new_borough_code', how="left")
df_1 = pd.merge(df_1, df_12.loc[(df_12["Year"] == 2011), ["new_borough_code", "rate of 5 year business survival"]], on='new_borough_code', how="left")
df_1.set_index("old_borough_code")
df_1 = pd.merge(df_1, df_13[["old_borough_code","prevalence of households with full employment", "prevalence of households with partial employment", "prevalence of households with no employment"]], on='old_borough_code', how="left")
df_1.set_index("new_borough_code")
df_1 = pd.merge(df_1, df_14.loc[(df_14["Year"] == "2015/16") & (df_14["Sex"] == "All"), ["new_borough_code", "average GCSE score"]], on='new_borough_code', how="left")

# Clean data to remove symbols blocking dtype conversion
df_1.iloc[:,15] = [x.replace(',', '') for x in df_1.iloc[:,15]]
df_1.iloc[:,16] = [x+65 for x in df_1.iloc[:,16]]

# Convert all numeric columns to dtype float64
for col in range(4,len(df_1.columns)):
    df_1.iloc[:,col] = pd.to_numeric(df_1.iloc[:,col], errors='coerce').fillna(0).astype(float)

# Export merged data for perusing
df_1.to_csv('A03-export_merged.csv')

# Remove outliers (London City) before graphing
#df_1 = df_1[df_1.borough_name != 'City of London']
df_1_backup = df_1.copy()

################################

combo_x_axis = None
combo_y_axis = None

#################################

def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]
    
#################################

outlier_list =  dict()
def toggle_outliers(s):
    global df_1
    global df_1_backup
    df_1 = df_1_backup.copy()
    global checkbutton_list
    if s in outlier_list:
        del outlier_list[s]
    else:
        outlier_list[s] = True
    for key, val in outlier_list.iteritems():
        df_1 = df_1[df_1.borough_name != key]

#################################

def graph_x_y():
    plt.close()
    fig = plt.gcf()
    ax = plt.gca()
    
    # Dropping NaN's is required to use numpy's polyfit
    df1_subset = df_1.dropna(subset=[combo_x_axis.get(), combo_y_axis.get()])
    
    # Set axis labels
    plt.xlabel(combo_x_axis.get(),fontsize=10)
    plt.ylabel(combo_y_axis.get(),fontsize=10)
    
    # Fit a linear trend line
    x = df1_subset[combo_x_axis.get()]
    y = df1_subset[combo_y_axis.get()]
    poly_coeff = np.polyfit(x, y, 1)
    poly_coeff_1d = np.poly1d(poly_coeff)
    #plt.plot(np.unique(x), poly_coeff_1d (np.unique(x)), color='black', linestyle='--', label='OLS linear regression line')
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    line = slope * x + intercept
    plt.plot(x, line, color='black', linestyle='--', label='OLS linear regression line', dashes=(1, 5))
    
    # Plot ellipse 2 standard deviations from mean of observed values
    num_std_dev = 2
    cov = np.cov(df1_subset[combo_x_axis.get()], df_1[combo_y_axis.get()])
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    ell_w, ell_h = 2 * num_std_dev * np.sqrt(vals)
    std_y = np.std(y)  #ell_h // 2
    ell_center = (np.mean(df_1[combo_x_axis.get()]), np.mean(df_1[combo_y_axis.get()]))
    ell = Ellipse(xy=ell_center, width=ell_w, height=ell_h, angle=theta, color='black', )
    ax.add_patch(ell)
    ell.set_facecolor('none')
    ax.add_artist(ell)
    plt.plot(np.NaN, np.NaN, color='k', label=r'$2 \sigma$ from data (95%)')
    
    # Plot lines 2 standard deviations from regression line
    plt.plot(x, line+(std_y*2), c='y', linestyle='-', label='_nolegend_')
    plt.plot(x, line-(std_y*2), c='y', linestyle='-', label=r'$2 \sigma$ from regression line (95%)')
    
    # Plot (x,y) points
    #df_1.plot(x=combo_x_axis.get(), y=combo_y_axis.get(), kind='scatter')
    x_middle = np.mean([ax.get_xlim()])
    y_middle = np.mean([ax.get_ylim()])
    for i, row in df_1[["borough_name", combo_x_axis.get(), combo_y_axis.get()]].iterrows():
        point_label = row[0]
        x_i = row[1]
        y_i = row[2]
        if y_i < ((slope * x_i + intercept)+(std_y*2)) and y_i > ((slope * x_i + intercept)-(std_y*2)):
        	plt.scatter(x_i,y_i)
        else:
            if (x_i > x_middle) and (y_i > y_middle):
                plt.annotate( "%s" %str(point_label), xy=(x_i,y_i), xytext=(0,-3), ha='right', va="top", textcoords='offset points')
            if (x_i > x_middle) and (y_i < y_middle):
                plt.annotate( "%s" %str(point_label), xy=(x_i,y_i), xytext=(0,3), ha='right', va="bottom", textcoords='offset points')
            if (x_i < x_middle) and (y_i > y_middle):
                plt.annotate( "%s" %str(point_label), xy=(x_i,y_i), xytext=(0,-3), ha='left', va="top", textcoords='offset points')
            if (x_i < x_middle) and (y_i < y_middle):
                plt.annotate( "%s" %str(point_label), xy=(x_i,y_i), xytext=(0,3), ha='left', va="bottom", textcoords='offset points')
            #plt.text(x_i+x_offset, y_i+y_offset, point_label)
            plt.scatter(x_i,y_i, c='k')
            print x_i,",",y_i
    
    # Add legend
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,ncol=2, borderaxespad=0.)
    
    plt.figtext(0.5,0.05,"Fig. 2 - Ordinary least squares linear regression", horizontalalignment='center', multialignment='center')
    fig.dpi=100
    fig.set_size_inches(8, 6)
    plt.tight_layout(w_pad=1, h_pad=1, pad=5)
    
    plt.show()
    #restore_outliers()

#################################

def steps(L):
    delta = 1.0/len(L)
    y = 0
    rL = list()
    for x in L:
        y += delta
        rL.append(y)
    return rL

def test_normality_graph():
    plt.close()
    fig = plt.gcf()
    grid = plt.GridSpec(4, 8, wspace=0.3, hspace=0.75)
    ax0 = plt.subplot(grid[:-1, 0:4])
    ax1 = plt.subplot(grid[-1, 0:4])
    ax2 = plt.subplot(grid[0:3, 5:8])#scatterplot and curve of residual error fit
    ax3 = plt.subplot(grid[-1, 5:8])#horizontal line chart showing residuals
    
    # Data
    data = pd.DataFrame({combo_x_axis.get(): df_1[combo_x_axis.get()], 'borough_name':df_1['borough_name'] })
    data = data.sort_values(by=[combo_x_axis.get()])
    data_mean = df_1[combo_x_axis.get()].mean()
    data["residual"] = df_1[combo_x_axis.get()] - data_mean
    res_mean = data["residual"].mean()
    res_std_dev = data["residual"].std(ddof=0)
    data["z_score_of_res"] = (data["residual"] - res_mean) / res_std_dev
    
    # Calculate normal distribution curve
    (mu, sigma) = norm.fit(data["z_score_of_res"])
    
    # Plot histogram
    n, bins, patches = ax0.hist(data["z_score_of_res"], bins=60, density=True, facecolor='white', alpha=0.75, label="Data", stacked=True, ec='black')
    
    # Plot normal distribution curve
    y = norm.pdf(bins, mu, sigma)
    l = ax0.plot(bins, y, 'b--', linewidth=2, label="fitted normal curve")
    
    # Set axis labels
    ax0.set(xlabel='Z-Score of residual', ylabel="Probability density")
    
    # Add legend
    ax0.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    
    # Add box whisker plot
    outlier_marker = dict(marker='s', markerfacecolor='white')
    mean_marker = dict(marker='s', markerfacecolor='white')
    meanlineprops = dict(linestyle='--', linewidth=0.8, color='red')
    medianlineprops = dict(linestyle='-', linewidth=0.8, color='blue')
    ax1.boxplot(data["z_score_of_res"], flierprops=outlier_marker, vert=False, showmeans=True, meanprops=meanlineprops, meanline=True, medianprops=medianlineprops)
    ax1.plot(np.NaN, np.NaN, 's', color='black',  markerfacecolor='white', label="outliers")
    ax1.plot(np.NaN, np.NaN, color='orange', label="median")
    ax1.plot(np.NaN, np.NaN, '--', color='red', label="mean")
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncol=3)
    ax1.tick_params(axis='both', left=False, labelleft=False)
    for i, row in data.iterrows():
    	point_label = row["borough_name"]
        if (row["z_score_of_res"] < -2.698):
            ax1.annotate( "%s" %str(point_label), xy=(row["z_score_of_res"],1), xytext=(0,-4), ha='left', va="top", textcoords='offset points')
        if (row["z_score_of_res"] > 2.698):
            ax1.annotate( "%s" %str(point_label), xy=(row["z_score_of_res"],1), xytext=(0,-4), ha='right', va="top", textcoords='offset points')
    ax1.set(xlabel='Z-Score of residual', ylabel="")
    
    #https://telliott99.blogspot.com/2012/04/ks-test.html
    xL = data["z_score_of_res"].sort_values()
    D, p_val = kstest(xL,'norm')
    yL = steps(xL)
    
    
    # Cumulative distribution curve
    rv = norm()
    pL = np.linspace(-4, 4)
    ax2.plot(pL, rv.cdf(pL), '--', label="Normal cumulative density")
    
    # vertical lines
    dL = list()
    for x,y0 in zip(xL,yL):
        y1 = rv.cdf(x)
        dL.append(abs(y1-y0))
        ax2.plot((x,x),(y0,y1), color='k', zorder=0)
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-0.1, 1.1)
    
    ax2.scatter(xL, yL, s=50, zorder=3, facecolors='white', edgecolors='k', label="Normalised data")
    
    # Add to legend
    ax2.plot(np.NaN, np.NaN, color='k', label="Residual")
    ax2.set(xlabel="Z-Score of residual", ylabel="Cumulative probability")
    
    # Add legend
    ax2.legend(loc=0)
    
    # Add title
    ax2.set_title(r'$\mathrm{Kolmogorov-Smirnov\ test:}\ D=%.3f,\ p=%.3f$' %(D, p_val))
    
    # Add horizontal line
    #ax3.axhline(y=0,xmin=-4,xmax=4,c="blue",linewidth=0.5,zorder=0)
    #ax3.stem(data["z_score_of_res"], data["residual"], linefmt='-.', markerfmt='C6o', basefmt='C3-')
    (markers, stemlines, baseline) = ax3.stem(data["z_score_of_res"], data["residual"], '--')
    plt.setp(markers, marker='o', markeredgecolor="black", markerfacecolor="white", zorder=3)
    ax3.set(xlabel='Z-Score of residual', ylabel="Residual")
    
    plt.figtext(0.5,0.025,"Fig. 3 - Distribution of residuals for: " + combo_x_axis.get(), horizontalalignment='center', multialignment='center')
    fig.set_size_inches(16, 8)
    fig.dpi=80
    #plt.tight_layout(w_pad=1, h_pad=1, pad=5)
    plt.show()
#################################
def plot_corr(df,w_size,h_size):
    '''Plot a graphical correlation matrix for a dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''

    # Compute the correlation matrix for the received dataframe
    corr_df = df.corr(method='pearson')
    
    # Filter lower triangular section
    corr_df2 = np.tril(corr_df, k=0)
    
    # Create heatmap
    cmap = cm.get_cmap('bwr_r', lut=21)
    
    # Get Figure and axes objects
    ax = plt.gca()
    fig = plt.gcf()
    
    # Plot the correlation matrix
    im = ax.imshow(corr_df2, interpolation="none", cmap=cmap, vmin=-1, vmax=1)
    plt.colorbar(im, label='Pearson Correlation Coefficient (r)')
    ax.set_aspect('auto')
    ax.set_xticks(np.arange(len(corr_df.columns)))
    ax.set_yticks(np.arange(len(corr_df.columns)))
    ax.set_xticklabels(corr_df.columns)
    ax.set_yticklabels(corr_df.columns)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    
    # Loop over data dimensions and create text annotations.
    for i in range(len(corr_df2)):
        for j in range(len(corr_df2)):
            if (corr_df2[i, j] != 0.0 and corr_df2[i, j] == corr_df2[i, j]):
                text = ax.text(j, i, round(corr_df2[i, j],1), ha="center", va="center", color="k", fontsize=7)
    
    fig.set_size_inches(w_size, h_size)
    fig.dpi=90
    plt.xlabel("Fig. 1 - Cross-correlation matrix of London stastics by borough",fontsize=10)
    plt.tight_layout(w_pad=1, h_pad=1, pad=2)
    
    # Create Interactive Tk Window
    root = tk.Tk()
    root.wm_title("Embedding in TK")
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    
    def on_key_event(event):
        print('you pressed %s' % event.key)
        key_press_handler(event, canvas, toolbar)

    canvas.mpl_connect('key_press_event', on_key_event)
    
    def _quit():
        root.quit()
        root.destroy()
    
    # Add Quit buttom
    quit_button = tk.Button(master=root, text='Quit', command=_quit)
    quit_button.pack()
    
    # Add Graph buttom
    graph_button = tk.Button(master=root, text='Graph', command=graph_x_y)
    graph_button.pack()
    
    # Add Normality buttom
    norm_button = tk.Button(master=root, text='Test for normality (x-axis)', command=test_normality_graph)
    norm_button.pack()
    
    # x-axis graph dropdown menu
    frame1 = ttk.Frame(root)
    frame1.pack(fill=tk.X)
    x_label = ttk.Label(frame1, text="Select column for x-axis: ")
    x_label.pack(side=tk.LEFT, padx=5, pady=5)
    global combo_x_axis
    combo_x_axis = ttk.Combobox(frame1, width=40)
    combo_x_axis['values']= list(df_1.iloc[:,4:].columns.values)
    combo_x_axis.current(0)
    combo_x_axis.pack(fill=tk.X, padx=5, expand=True)
    
    # y-axis graph dropdown menu
    frame2 = ttk.Frame(root)
    frame2.pack(fill=tk.X)
    y_label = ttk.Label(frame2, text="Select column for y-axis: ")
    y_label.pack(side=tk.LEFT, padx=5, pady=5)
    global combo_y_axis
    combo_y_axis = ttk.Combobox(frame2, width=40)
    combo_y_axis['values']= list(df_1.iloc[:,4:].columns.values)
    combo_y_axis.current(0)
    combo_y_axis.pack(fill=tk.X, padx=5, expand=True)
    
    # Add scrollbox for removing outliers
    frame3 = ttk.Frame(root)
    frame3.pack(fill=tk.X)
    out_label = ttk.Label(frame3, text="Select outliers to remove: ")
    out_label.pack(side=tk.LEFT, padx=5, pady=5)
    vsb = tk.Scrollbar(frame3, orient="vertical")
    text = tk.Text(frame3, height=5, yscrollcommand=vsb.set, borderwidth=0)
    #vsb.config(command=root.text.yview)
    vsb.pack(side="right", fill="y")
    text.pack(side="left", fill="both", expand=True)
    
    global checkbutton_list
    for i, row in df_1[["borough_name"]].iterrows():
    	label = row[0]
        cb = tk.Checkbutton(frame3, text=label, command=lambda l=label: toggle_outliers(l))
        text.window_create("end", window=cb)
        text.insert("end", "\n") 
    
    tk.mainloop()
    
plot_corr(df_1, w_size=9, h_size=7)

exit()