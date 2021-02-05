# MINERVA
This is a Python 3 implementation of MINERVA, 
a series of memory systems developed since 1980s by Douglas L. Hintzman, coworkers and followers.

To deploy my implementation, the only thing you need is `minerva.py`. 
The dependencies are Python 3 and Numpy.

## MINERVA 2
I found some other GitHub repositories helpful (whereas most of the results when searching with 'MINERVA' are irrelevant), 
but none serves my own purpose. 
In addition, one only needs to deal with matrices and simple arithmetics, so I have decided to code my own.
One may want to compare this implementation with the following two, which contain some features that haven't been incorporated into this one:
1. https://github.com/dwhite54/minerva2
-- This is also a Python implementation. 
Other than differences in coding style, it chooses to use the entire length of a memory trace as the denominator when computing similarities,
while the original paper (Hintzman, 1984) presents a slightly different option.

2. https://github.com/deniztu/minerva_al
-- This is an R implementation.
It is seemingly meant for MINERVA-AL, but it contains MINERVA 2, too.
I do not choose it, simply because I am too lazy to import R into Python.
(I am working on a bigger project in which I need to compare different memory models, including MINERVA, 
so it seems to me more natural to use Python to pack everything belonging to a model, like a memory matrix, inside a class.)

I have made a primary performance comparison between my implementation and the above two. 
Please refer to [`implementation_comparison.ipynb`](https://github.com/anish-lu-yihe/MINERVA/blob/main/implementation_comparison.ipynb) for the details.

## References
1. MINERVA 2: [Hintzman (1984)](https://link.springer.com/article/10.3758/BF03202365)
2. MINERVA-AL: (t.b.c) 
3. MINERVA-DE: (t.b.c)