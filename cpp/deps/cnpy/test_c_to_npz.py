import sys
import numpy as np



if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("[ERROR]: I need the input filename")
        exit(1)

    input_filename = sys.argv[1]

    try:
        npz = np.loadz(input_filename)
    except:
        print(f"[ERROR]: the file `{input_filename}` is not .npz")
        exit(1)

    # here I'm assuming the structure of the dictionary
    assert('integers' in npz)
    assert('doubles' in npz)

    print(npz.keys())
    integers = npz['integers']
    doubles = npz['doubles']

    ok = True

    if not np.allclose(integers, np.arange(2, 10)):
        print(f"Expected integers from 1 to 10, got {integers}")
        ok = False

    if not np.allclose(doubles, np.arange(1, 10, dtype=np.float64)):
        print(f"Expected doubles from 1.0 to 10.0, got {integers}")
        ok = False

    exit(0 if ok else 1)
