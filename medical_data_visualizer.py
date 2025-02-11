import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
bmi = df['weight'] / np.square(df['height']/100)
df['overweight'] = (bmi > 25).astype('uint8')
# 3
df['gluc'] = (df['gluc'] != 1).astype('uint8')
df['cholesterol'] = (df['cholesterol'] != 1).astype('uint8')
# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')    

    # 7
    fig = sns.catplot(x="variable", y="total", hue="value", col="cardio", data=df_cat, kind="bar").fig

    # 8

    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &  # Systolic >= Diastolic
        (df['height'] >= df['height'].quantile(0.025)) &  # Remove outliers (height < 2.5%)
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &  # Remove outliers (weight < 2.5%)
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))


    # 14
    fig, ax = plt.subplots(figsize=(12, 8))  # Create figure and axes

    # 15

    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", cmap="coolwarm", linewidths=0.5, ax=ax)  # Plot on ax

    # 16
    fig.savefig('heatmap.png')
    return fig

