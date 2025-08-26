import os


def save_table(df, saving_path, name, format=['csv']):
    df.to_csv(os.path.join(saving_path, f'{name}.csv'))


def df_to_latex(df, filename, caption, label, form=None):
    """
       Export a pandas DataFrame to a LaTeX file:
       - Landscape orientation
       - Scaled to fit one page width
       - Arial font, 10pt
       - Single caption + label
       """
    # Generate only the tabular environment (no caption, no label)
    if form is None:
        latex_table = df.to_latex(
            index=False,
            longtable=False,
            escape=False,
        )
    else:
        latex_table = df.to_latex(
            index=False,
            longtable=False,
            escape=False,
            formatters=form
        )

    # Wrap in full LaTeX document
    latex_doc = r"""
    \documentclass[a4paper,10pt]{article} % 10pt font
    \usepackage{booktabs}    % professional table lines
    \usepackage{adjustbox}    % scale table to width
    \usepackage{lscape}       % landscape pages
    \usepackage{caption}      % table captions
    \usepackage{helvet}       % Arial font
    \renewcommand{\familydefault}{\sfdefault}  % set default font to sans serif

    \begin{document}
    \begin{landscape}
    \begin{table}[htbp]
    \centering
    \caption{""" + caption + r"""}
    \label{""" + label + r"""}
    \begin{adjustbox}{width=\textwidth}
    """ + latex_table + r"""
    \end{adjustbox}
    \end{table}
    \end{landscape}
    \end{document}
    """

    with open(filename, "w") as f:
        f.write(latex_doc)
