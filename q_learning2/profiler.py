import pstats

p = pstats.Stats('make_action.profile')
p.sort_stats('cumulative').print_stats(10)  # Show top 10 functions sorted by 'cumulative' time
