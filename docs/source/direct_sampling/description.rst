.. currentmodule:: pygauss.direct_sampling

.. _direct_sampling_description:

Description
###########

This Python module implements existing approaches, directly derived from numerical linear algebra, to sample from  high-dimensional Gaussian probability distributions. The latter can be divided into three groups, namely:

- factorization approaches (e.g., Cholesky or square-root samplers),
- square-root approximation approaches (e.g., Chebyshev and Lanczos samplers),
- conjugate-gradient samplers.

For more details, we refer the interested reader to Section 3 of the companion paper.