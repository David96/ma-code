# Code for my master thesis

## Usage (fixed_disks)

## InitialSpacings.jl

Used to find spacings with all disks being free. Runs in two stages:

1. Find equidistant spacing with peak and write to init_\$(freq).txt (this should be checked manually, sometimes the 
   range which is searched has to be adjusted)
2. Find an optimized spacing for the broadband boostfactor.

### Running

Currently the frequencies for which the spacings should be found have to be manually changed in the for loop. Afterwards 
it can simply be run from the command line.

## freq_scan.jl

Takes a frequency as the parameter for which it optimizes the spacing with one disk fixed (runs for every disk).

Suggested way of running is in a for loop like so:

```sh
for i in `seq 50e6 50e6 1500e6`; do julia ShiftFreqs.jl $i; done
```

This runs for shifts from 50e6 to 1500e6 with 50e6 steps. F0 has to be set manually in the file.

## plot_stuff.jl

This has several methods for creating interesting plots. Note that the plot_bf_quality expects a file created by 
InitialSpacings.jl for every shift to normalize the shifted boostfactor with a fixed disk.

## FileUtils.jl

Note the prefix variable in the beginning - the maxwell cluster made some problems when saving stuff in the user 
directory so when running on a cluster change that to your beegfs directory (or wherever you want to save your results).
