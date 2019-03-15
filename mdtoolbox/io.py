import re
import pandas as pd

def xvg2df(path):
    """ Reads input data from an xvg file into a pandas dataframe """
    header_rows = 0
    col_names = []
    index_col = 0
    with open(path, 'r') as file:
        for line in file.readlines():
            if line.startswith("#"):
                header_rows += 1
                continue
            if line.startswith("@"):
                header_rows += 1
                try:
                    col_name = re.search('\@ s[0-9]* legend \"(.*)\"',
                                         line).group(1)
                    col_names.append(col_name.replace(" ", "_"))
                except Exception:
                    pass
    print(header_rows)
    print(col_names)
    print(index_col)
    return pd.read_csv(path, delim_whitespace=True, header=header_rows,
                       index_col=index_col, names=col_names)

