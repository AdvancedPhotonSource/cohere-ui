import cohere_scripts.inner_scripts.te_rec as te_rec
import time


if __name__ == "__main__":
    st = time.time()
    # running on hpc
    hpc = False
    exit_code = te_rec.time_evolving_rec(hpc)
    en = time.time()
    print(f'reconstruction took {en - st} seconds.')
    exit(exit_code)

# mpiexec -n 16 python hpc_scripts/te_rec.py <experiment_dir>