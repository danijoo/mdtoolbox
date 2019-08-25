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


def xtc_iterator(structure_file, trajectory_file, start=0, stop=None, step=None, batch_size=1000):
    """ Allows to iterate over a trajectory in a memory friendly but less efficient way: it loads the trajectory
    in chunks of batch_size and yields single frames from this batch until the batch is done.
    Then, the moves to the next batch.
    
    Yields a tuple containing (timestep, AtomArray)
    """
    if step is None:
        step = 1
    if stop is None:
        stop = np.inf
    structure = io.load_structure(structure_file)
    
    batch_start = start
    while batch_start < stop:  # iterate over batches
        batch_stop = batch_start + batch_size*step
        if batch_stop > stop:
            batch_stop = stop
#         print("start", batch_start, "stop", batch_stop, "step", step)
        f = xtc.XTCFile()
        f.read(trajectory_file, start=batch_start, stop=batch_stop, step=step)
        time = f.get_time()
        trj = f.get_structure(structure)
        if len(trj) == 0:  # end of trj reached
            return
        for ts, frame in zip(time, trj):  # iterate over frames in batch
            yield (ts, frame)
        batch_start += batch_size*step
        if batch_start > stop:
            return
        