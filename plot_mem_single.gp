set title "Memory Usage vs Time"
set xlabel "Time (s)"
set ylabel "Memory (MB)"

# Major ticks every 600 seconds (10 minutes)
set xtics 600
# Minor ticks every 60 seconds (1 minute)
set mxtics 10  # 600/60 = 10 subdivisions

# Optional: Improve grid visibility
set grid xtics mxtics linetype 1 linecolor "gray"

# Plot using row number as time (if no timestamp)
plot "memory_usage_6k_fix1.log" using 0:($3/1024) with lines title "6K batch size fix1"

# OR if your log has timestamps in column 1:
# plot "memory_usage.log" using 1:($3/1024) with lines title "RSS Memory"

