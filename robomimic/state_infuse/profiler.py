import pstats

p = pstats.Stats("train_batch.prof")
p.strip_dirs().sort_stats("cumulative").print_stats(20)


