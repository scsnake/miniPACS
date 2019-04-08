import multiprocessing as mp
from glob import glob
from pathlib import Path

import pydicom


# random.seed(123)


# define a example function
def load_dicom(path, output):
    """ Generates a random string of numbers, lower- and uppercase chars. """
    d = pydicom.dcmread(path).pixel_array
    output.put(d)


if __name__ == '__main__':
    # Define an output queue
    output = mp.Queue()
    dir = r'C:\f8ecf6be8ae631c6dd694c9638a02b45'
    # Setup a list of processes that we want to run
    processes = [mp.Process(target=load_dicom, args=(p, output)) for p in glob(str(Path(dir).joinpath('*.dcm')))[0:2]]

    # Run processes
    for p in processes:
        p.start()

    # Exit the completed processes
    for p in processes:
        p.join()

    # Get process results from the output queue
    results = [output.get() for p in processes]

    print(results)
