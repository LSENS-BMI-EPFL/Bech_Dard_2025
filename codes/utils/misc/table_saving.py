import os


def save_table(df, saving_path, name, format=['csv']):
    df.to_csv(os.path.join(saving_path, f'{name}.csv'))

