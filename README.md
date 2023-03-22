# dg-swe-code
Code for the paper "Conservation and stability in a discontinuous Galerkin method for the vector invariant spherical shallow water equations"

## Installation

``pip install .``

## Running experiments

The experiments are located in the `swe-experiments` folder. Except for the Galewsky 
test case, all test cases as they appear in the paper can simply be run with python. 
The results will appear in `swe-experiments/plots`.

### Galewsky test case

For efficiency reason the Galewsky test case is configured at a lower resolution of 
`6x15x15` elements. To change this simply set `nx=ny=...` at the top of the file 
to your desired resolution. To run the test case set the `mode=run`, then to plot the results set `mode=plot`.

