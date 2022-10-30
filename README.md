# primal simplex method

## Dependencies

- `python3`, version >= 3.10

  - `pip` and `venv` (just to containerize the numpy dependency)

## Instructions

Clone the repo and do:

```sh
python3 -m venv .env
source ./.env/bin/activate
pip install -r requirements.txt
python3 main.py
```

And then manually insert the data for the linear optimization problem, which should be in the form

$min \ c^Tx$

$s.t. \ Ax = b$

$\ \ \ \ \ \  \ \ x\geq0$

The first input is the number of constraints $m$, the second is $n$ (the number of decision varibles in $x$). Then you
should input the elements in the costs vector $c$ and in the resources vector $b$ separated by single spaces. After
that, you are prompted to input the matrix $A$, which should be inputted one line at a time (with elements separated by
single spaces, and lines themselves separated by EOL). At last, input the maximum number of iterations you want the code
to execute, and wait for the results. After that, you also will have the optionality of solving another problem without
quitting the run session and having to restart.

If you do not want to run each problem at a time, open `test.py` and add all your data there (according to the examples
there), and then

```sh
python3 test.py
```

## Notes

This is a uni project with the report in PTBR called `Projeto Computacional - MS428.pdf`, so it was *not* made for large scale testing, thus
we did not implement a way of reading data from files (but it is doable, and the algorithm should work -- although I do
not make promises about its performance, because this is python, come on!)

