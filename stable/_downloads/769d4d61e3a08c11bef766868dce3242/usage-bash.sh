# Usage - command line
# ====================
#
# Scatterkit can be used directly from the command line (cli). Using cli instead
# of a Jupyter notebook can sometimes be more comfortable, particularly for
# lengthy analysis. The cli in particular is handy because it allows for
# updating the analysis results during the run. You can specify the number of
# frames after the output is updated with the ``-concfreq`` flag. See below for
# details.
#
# Note that in this documentation, we almost exclusively describe the use of scatterkit from
# the python interpreter, but all operations can be equivalently performed from the cli.

scatterkit saxs -s water_nvt.tpr \
                -f water_nvt.xtc \
                -atomgroup 'resname SOL'

# %%
#
# The SAXS profile has been written in a file named ``sq.dat`` in the current
# directory. The written file starts with the following lines

head -n 20 sq.dat

# %%
#
# For lengthy analysis, use the ``concfreq`` option to update the result during the run

scatterkit saxs -s water_nvt.tpr \
                -f water_nvt.xtc \
                -atomgroup 'resname SOL' \
                -concfreq '10'

# %%
#
# The general help of scatterkit can be accessed using

scatterkit -h

# %%
#
# Package-specific page can also be accessed from the cli

scatterkit saxs -h
