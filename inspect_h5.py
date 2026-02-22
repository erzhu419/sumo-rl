import h5py
import sys

def inspect_file(filename):
    print(f"--- Inspecting {filename} ---")
    try:
        with h5py.File(filename, 'r') as f:
            for key in f.keys():
                dset = f[key]
                print(f"Dataset: {key}")
                print(f"  Shape: {dset.shape}")
                print(f"  Dtype: {dset.dtype}")
                print(f"  Compression: {dset.compression}")
                print(f"  Compression Opts: {dset.compression_opts}")
    except Exception as e:
        print(f"Error: {e}")
    print("\n")

if __name__ == "__main__":
    files = sys.argv[1:]
    for f in files:
        inspect_file(f)
