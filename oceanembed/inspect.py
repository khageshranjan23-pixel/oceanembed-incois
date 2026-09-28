import argparse
import xarray as xr

def main():
    p=argparse.ArgumentParser();p.add_argument('file');a=p.parse_args()
    with xr.open_dataset(a.file) as d:
        print(d)
        for name,v in d.variables.items():print(name,v.dims,dict(v.attrs))
if __name__=='__main__':main()
