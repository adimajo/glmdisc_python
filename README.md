[![PyPI version](https://badge.fury.io/py/glmdisc.svg)](https://badge.fury.io/py/glmdisc)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/glmdisc.svg)](https://pypi.python.org/pypi/glmdisc/)
[![PyPi Downloads](https://img.shields.io/pypi/dm/glmdisc)](https://img.shields.io/pypi/dm/glmdisc)
[![Build Status](https://travis-ci.org/adimajo/glmdisc_python.svg?branch=master)](https://travis-ci.org/adimajo/glmdisc_python)
![Python package](https://github.com/adimajo/glmdisc_python/workflows/Python%20package/badge.svg)
[![codecov](https://codecov.io/gh/adimajo/glmdisc_python/branch/master/graph/badge.svg)](https://codecov.io/gh/adimajo/glmdisc_python)

# Feature quantization for parsimonious and interpretable models

Table of Contents
-----------------

* [Documentation](https://adimajo.github.io/glmdisc_python)
* [Installation instructions](#-installing-the-package)
* [Theory](#-use-case-example)
* [Some examples](#-the-glmdisc-package)
* [Open an issue](https://github.com/adimajo/glmdisc_python/issues/new/choose)
* [Contribute](#-contribute)

## Motivation

Credit institutions are interested in the refunding probability of a loan given the applicant’s characteristics in order to assess the worthiness of the credit. For regulatory and interpretability reasons, the logistic regression is still widely used to learn this probability from the data. Although logistic regression handles naturally both quantitative and qualitative data, three pre-processing steps are usually performed: firstly, continuous features are discretized by assigning factor levels to pre-determined intervals; secondly, qualitative features, if they take numerous values, are grouped; thirdly, interactions (products between two different predictors) are sparsely introduced. By reinterpreting discretized (resp. grouped) features as latent variables, we are able, through the use of a Stochastic Expectation-Maximization (SEM) algorithm and a Gibbs sampler to find the best discretization (resp. grouping) scheme w.r.t. the logistic regression loss. For detecting interacting features, the same scheme is used by replacing the Gibbs sampler by a Metropolis-Hastings algorithm. The good performances of this approach are illustrated on simulated and real data from Credit Agricole Consumer Finance.

This repository is the implementation of [Ehrhardt Adrien, et al. "Feature quantization for parsimonious and interpretable predictive models." arXiv preprint arXiv:1903.08920 (2019)](https://arxiv.org/abs/1903.08920).

NOTE: for now, only "glmdisc-SEM" is available.

## Getting started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

This code is supported on Python 3.7, 3.8, 3.9 and 3.10 (see [tox file](tox.ini)).

### Installing the package

#### Installing the development version

If `git` is installed on your machine, you can use:

```PowerShell
pip install git+https://github.com/adimajo/glmdisc_python.git
```

If `git` is not installed, you can also use:

```PowerShell
pip install --upgrade https://github.com/adimajo/glmdisc_python/archive/master.tar.gz
```

#### Installing through the `pip` command

You can install a stable version from [PyPi](https://pypi.org/project/glmdisc/) by using:

```PowerShell
pip install glmdisc
```

#### Installation guide for Anaconda

The installation with the `pip` command **should** work. If not, please raise an issue.

#### For people behind proxy(ies)...

A lot of people, including myself, work behind a proxy at work...

A simple solution to get the package is to use the `--proxy` option of `pip`:

```PowerShell
pip --proxy=http://username:password@server:port install glmdisc
```

where *username*, *password*, *server* and *port* should be replaced by your own values.


**What follows is a quick introduction to the problem of discretization and how this package answers the question.**

<!--**If you wish to see the package in action, please refer to the accompanying Jupyter Notebook.**-->

<!--**If you seek specific assistance regarding the package or one of its function, please refer to the ReadTheDocs.**-->

## Use case example

For a thorough exaplanation of the approach, see [this blog post](https://adimajo.github.io/discretization) or [this article](https://arxiv.org/abs/1903.08920).

If you're interested in directly using the package, you can skip this part and go to [this part below](#-the-glmdisc-package).

In practice, the statistical modeler has historical data about each customer's characteristics. For obvious reasons, only data available at the time of inquiry must be used to build a future application scorecard. Those data often take the form of a well-structured table with one line per client alongside their performance (did they pay back their loan or not?) as can be seen in the following table:

| Job | Habitation | Time in job | Children | Family status | Default |
| --- | --- | --- | --- | --- | --- |
| Craftsman | Owner | 10 | 0 | Divorced |  No |
| Technician | Renter | **Missing** | 1 | Widower | No |
| **Missing** | Starter | 5 | 2 | Single |  Yes |
| Office employee | By family | 2 | 3 | Married | No |

## Notations

In the rest of the vignette, the random vector <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;X=(X_j)_1^d" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;X=(X_j)_1^d" title="X=(X_j)_1^d" /></a>  will designate the predictive features, i.e. the characteristics of a client. The random variable <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;Y&space;\in&space;\{0,1\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;Y&space;\in&space;\{0,1\}" title="Y \in \{0,1\}" /></a>  will designate the label, i.e. if the client has defaulted (<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;Y=1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;Y=1" title="Y=1" /></a>) or not (<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;Y=0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;Y=0" title="Y=0" /></a>).

We are provided with an i.i.d. sample <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;(\mathbf{x},\mathbf{y})&space;=&space;(x_i,y_i)_1^n" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;(\mathbf{x},\mathbf{y})&space;=&space;(x_i,y_i)_1^n" title="(\mathbf{x},\mathbf{y}) = (x_i,y_i)_1^n" /></a> consisting in <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;n" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;n" title="n" /></a> observations of <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;X" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;X" title="X" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;Y" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;Y" title="Y" /></a>.

## Logistic regression

The logistic regression model assumes the following relation between <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;X" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;X" title="X" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;Y" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;Y" title="Y" /></a> :

<a href="https://www.codecogs.com/eqnedit.php?latex=\ln&space;\left(&space;\frac{p_\theta(Y=1|x)}{p_\theta(Y=0|x)}&space;\right)&space;=&space;\theta_0&space;&plus;&space;\sum_{j&space;\text{&space;if&space;}&space;X_j&space;\text{&space;continuous}}&space;\theta_j&space;x_j&space;&plus;&space;\sum_{j&space;\text{&space;if&space;}&space;X_j&space;\text{&space;categorical}}&space;\theta_j^{x_j}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\ln&space;\left(&space;\frac{p_\theta(Y=1|x)}{p_\theta(Y=0|x)}&space;\right)&space;=&space;\theta_0&space;&plus;&space;\sum_{j&space;\text{&space;if&space;}&space;X_j&space;\text{&space;continuous}}&space;\theta_j&space;x_j&space;&plus;&space;\sum_{j&space;\text{&space;if&space;}&space;X_j&space;\text{&space;categorical}}&space;\theta_j^{x_j}" title="\ln \left( \frac{p_\theta(Y=1|x)}{p_\theta(Y=0|x)} \right) = \theta_0 + \sum_{j \text{ if } X_j \text{ continuous}} \theta_j x_j + \sum_{j \text{ if } X_j \text{ categorical}} \theta_j^{x_j}" /></a>

where <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\theta&space;=&space;(\theta_j)_0^d" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\theta&space;=&space;(\theta_j)_0^d" title="\theta = (\theta_j)_0^d" /></a> are estimated using <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;(\mathbf{x},\mathbf{y})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;(\mathbf{x},\mathbf{y})" title="(\mathbf{x},\mathbf{y})" /></a> (and <a href="https://www.codecogs.com/eqnedit.php?latex=\theta_j^h,&space;1&space;\leq&space;h&space;\leq&space;l_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta_j^h,&space;1&space;\leq&space;h&space;\leq&space;l_j" title="\theta_j^h, 1 \leq h \leq l_j" /></a> denotes the coefficients associated with a categorical feature <a href="https://www.codecogs.com/eqnedit.php?latex=x_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x_j" title="x_j" /></a> being equal to <a href="https://www.codecogs.com/eqnedit.php?latex=h" target="_blank"><img src="https://latex.codecogs.com/gif.latex?h" title="h" /></a>).

Clearly, for continuous features, the model assumes linearity of the logit transform of the response <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;Y" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;Y" title="Y" /></a> with respect to <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;X" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;X" title="X" /></a>.
On the contrary, for categorical features, it might overfit if there are lots of levels (<a href="https://www.codecogs.com/eqnedit.php?latex=l_j&space;>>&space;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?l_j&space;>>&space;1" title="l_j >> 1" /></a>). It does not handle missing values. 

## Common problems with logistic regression on "raw" data

Fitting a logistic regression model on "raw" data presents several problems, among which some are tackled here.

### Feature selection

First, among all collected information on individuals, some are irrelevant for predicting <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;Y" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;Y" title="Y" /></a>. Their coefficient <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\theta_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\theta_j" title="\theta_j" /></a> should be 0  which might (eventually) be the case asymptotically (i.e. <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;n&space;\rightarrow&space;\infty" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;n&space;\rightarrow&space;\infty" title="n \rightarrow \infty" /></a>).

Second, some collected information are highly correlated and affect each other's coefficient estimation.

As a consequence, data scientists often perform feature selection before training a machine learning algorithm such as logistic regression.

There already exists methods and packages to perform feature selection, see for example the `feature_selection` submodule in the `sklearn` package.

`glmdisc` is not a feature selection tool but acts as such as a side-effect: when a continuous feature is discretized into only one interval, or when a categorical feature is regrouped into only one value, then this feature gets out of the model.

For a thorough reference on feature selection, see e.g. Guyon, I., & Elisseeff, A. (2003). An introduction to variable and feature selection. *Journal of machine learning research, 3*(Mar), 1157-1182.

### Linearity

When provided with continuous features, the logistic regression model assumes linearity of the logit transform of the response <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;Y" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;Y" title="Y" /></a> with respect to <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;X" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;X" title="X" /></a>. This might not be the case at all.

For example, we can simulate a logistic model with an arbitrary power of <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;X" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;X" title="X" /></a> and then try to fit a linear logistic model:

```python


```
- [ ] Show the Python code

- [ ] Get this graph online

Of course, providing the `sklearn.linear_model.LogisticRegression` function with a dataset containing <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;X^5" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;X^5" title="X^5" /></a> would solve the problem. This can't be done in practice for two reasons: first, it is too time-consuming to examine all features and candidate polynomials; second, we lose the interpretability of the logistic decision function which was of primary interest.

Consequently, we wish to discretize the input variable <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;X" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;X" title="X" /></a> into a categorical feature which will "minimize" the error with respect to the "true" underlying relation:

- [ ] Show the Python code

- [ ] Get this graph online


### Too many values per categorical feature

When provided with categorical features, the logistic regression model fits a coefficient for all its values (except one which is taken as a reference). A common problem arises when there are too many values as each value will be taken by a small number of observations <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;x_i^j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;x_i^j" title="x_i^j" /></a> which makes the estimation of a logistic regression coefficient unstable:


- [ ] Show the Python code

- [ ] Get this graph online


If we divide the training set in 10 and estimate the variance of each coefficient, we get:

- [ ] Show the Python code

- [ ] Get this graph online



All intervals crossing 0 are non-significant! We should group factor values to get a stable estimation and (hopefully) significant coefficient values.


# Discretization and grouping: theoretical background

## Notations

Let <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;E=(\mathfrak{q}_j)_1^d" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;E=(\mathfrak{q}_j)_1^d" title="E=(\mathfrak{q}_j)_1^d" /></a> be the latent discretized transform of <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;X" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;X" title="X" /></a>, i.e. taking values in <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\{0,\ldots,m_j\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\{0,\ldots,m_j\}" title="\{0,\ldots,m_j\}" /></a> where the number of values of each covariate <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;m_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;m_j" title="m_j" /></a> is also latent.

The fitted logistic regression model is now:
<a href="https://www.codecogs.com/eqnedit.php?latex=\ln&space;\left(&space;\frac{p_\theta(Y=1|e)}{p_\theta(Y=0|e)}&space;\right)&space;=&space;\theta_0&space;&plus;&space;\sum_{j=1}^d&space;\sum_{k=1}^{m_j}&space;\theta^j_k*{1}_{e^j=k}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\ln&space;\left(&space;\frac{p_\theta(Y=1|e)}{p_\theta(Y=0|e)}&space;\right)&space;=&space;\theta_0&space;&plus;&space;\sum_{j=1}^d&space;\sum_{k=1}^{m_j}&space;\theta^j_k*{1}_{e^j=k}" title="\ln \left( \frac{p_\theta(Y=1|e)}{p_\theta(Y=0|e)} \right) = \theta_0 + \sum_{j=1}^d \sum_{k=1}^{m_j} \theta^j_k*{1}_{e^j=k}" /></a>

Clearly, the number of parameters has grown which allows for flexible approximation of the true underlying model <a href="https://www.codecogs.com/eqnedit.php?latex=p(Y|E)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(Y|E)" title="p(Y|E)" /></a>.

## Best discretization?

Our goal is to obtain the model <a href="https://www.codecogs.com/eqnedit.php?latex=p_\theta(Y|e)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p_\theta(Y|e)" title="p_\theta(Y|e)" /></a> with best predictive power. As <a href="https://www.codecogs.com/eqnedit.php?latex=E" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E" title="E" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=\theta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\theta" title="\theta" /></a> are both optimized, a formal goodness-of-fit criterion could be:
<a href="https://www.codecogs.com/eqnedit.php?latex=(\hat{\theta},\hat{\mathbf{e}})&space;=&space;\arg&space;\max_{\theta,\mathbf{e}}&space;\text{AIC}(p_\theta(\mathbf{y}|\mathbf{e}))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(\hat{\theta},\hat{\mathbf{e}})&space;=&space;\arg&space;\max_{\theta,\mathbf{e}}&space;\text{AIC}(p_\theta(\mathbf{y}|\mathbf{e}))" title="(\hat{\theta},\hat{\mathbf{e}}) = \arg \max_{\theta,\mathbf{e}} \text{AIC}(p_\theta(\mathbf{y}|\mathbf{e}))" /></a>
where AIC stands for Akaike Information Criterion.


## Combinatorics

The problem seems well-posed: if we were able to generate all discretization schemes transforming <a href="https://www.codecogs.com/eqnedit.php?latex=X" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X" title="X" /></a> to <a href="https://www.codecogs.com/eqnedit.php?latex=E" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E" title="E" /></a>, learn <a href="https://www.codecogs.com/eqnedit.php?latex=p_\theta(y|e)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p_\theta(y|e)" title="p_\theta(y|e)" /></a> for each of them and compare their AIC values, the problem would be solved.

Unfortunately, there are way too many candidates to follow this procedure. Suppose we want to construct k intervals of <a href="https://www.codecogs.com/eqnedit.php?latex=\mathfrak{q}_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathfrak{q}_j" title="\mathfrak{q}_j" /></a> given n distinct <a href="https://www.codecogs.com/eqnedit.php?latex=(x_j_i)_1^n" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(x_j_i)_1^n" title="(x_j_i)_1^n" /></a>. There is <a href="https://www.codecogs.com/eqnedit.php?latex=n&space;\choose&space;k" target="_blank"><img src="https://latex.codecogs.com/gif.latex?n&space;\choose&space;k" title="n \choose k" /></a> models. The true value of k is unknown, so it must be looped over. Finally, as logistic regression is a multivariate model, the discretization of <a href="https://www.codecogs.com/eqnedit.php?latex=\mathfrak{q}_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathfrak{q}_j" title="\mathfrak{q}_j" /></a> can influence the discretization of <a href="https://www.codecogs.com/eqnedit.php?latex=E^k" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E^k" title="E^k" /></a>, <a href="https://www.codecogs.com/eqnedit.php?latex=k&space;\neq&space;j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?k&space;\neq&space;j" title="k \neq j" /></a>.

As a consequence, existing approaches to discretization (in particular discretization of continuous attributes) rely on strong assumptions to simplify the search of good candidates as can be seen in the review of Ramírez‐Gallego, S. et al. (2016) - see References section.



# Discretization and grouping: estimation

## Likelihood estimation

<a href="https://www.codecogs.com/eqnedit.php?latex=E" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E" title="E" /></a> can be introduced in <a href="https://www.codecogs.com/eqnedit.php?latex=p(Y|X)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(Y|X)" title="p(Y|X)" /></a>:
<a href="https://www.codecogs.com/eqnedit.php?latex=\forall&space;\:&space;x,y,&space;\;&space;p(y|x)&space;=&space;\sum_e&space;p(y|x,e)p(e|x)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\forall&space;\:&space;x,y,&space;\;&space;p(y|x)&space;=&space;\sum_e&space;p(y|x,e)p(e|x)" title="\forall \: x,y, \; p(y|x) = \sum_e p(y|x,e)p(e|x)" /></a>

First, we assume that all information about <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;Y" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;Y" title="Y" /></a> in <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;X" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;X" title="X" /></a> is already contained in <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;E" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;E" title="E" /></a> so that:
<a href="https://www.codecogs.com/eqnedit.php?latex=\forall&space;\:&space;x,y,e,&space;\;&space;p(y|x,e)=p(y|e)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\forall&space;\:&space;x,y,e,&space;\;&space;p(y|x,e)=p(y|e)" title="\forall \: x,y,e, \; p(y|x,e)=p(y|e)" /></a>
Second, we assume the conditional independence of <a href="https://www.codecogs.com/eqnedit.php?latex=\mathfrak{q}_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mathfrak{q}_j" title="\mathfrak{q}_j" /></a> given <a href="https://www.codecogs.com/eqnedit.php?latex=X_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X_j" title="X_j" /></a>, i.e. knowing <a href="https://www.codecogs.com/eqnedit.php?latex=X_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X_j" title="X_j" /></a>, the discretization <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\mathfrak{q}_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\mathfrak{q}_j" title="\mathfrak{q}_j" /></a> is independent of the other features <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;X^k" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;X^k" title="X^k" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;E^k" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;E^k" title="E^k" /></a> for all <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;k&space;\neq&space;j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;k&space;\neq&space;j" title="k \neq j" /></a>:
<a href="https://www.codecogs.com/eqnedit.php?latex=\forall&space;\:x,&space;k\neq&space;j,&space;\;&space;\mathfrak{q}_j&space;|&space;x_j&space;\perp&space;E^k&space;|&space;x^k" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\forall&space;\:x,&space;k\neq&space;j,&space;\;&space;\mathfrak{q}_j&space;|&space;x_j&space;\perp&space;E^k&space;|&space;x^k" title="\forall \:x, k\neq j, \; \mathfrak{q}_j | x_j \perp E^k | x^k" /></a>
The first equation becomes:
<a href="https://www.codecogs.com/eqnedit.php?latex=\forall&space;\:&space;x,y,&space;\;&space;p(y|x)&space;=&space;\sum_e&space;p(y|e)&space;\prod_{j=1}^d&space;p(e^j|x_j)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\forall&space;\:&space;x,y,&space;\;&space;p(y|x)&space;=&space;\sum_e&space;p(y|e)&space;\prod_{j=1}^d&space;p(e^j|x_j)" title="\forall \: x,y, \; p(y|x) = \sum_e p(y|e) \prod_{j=1}^d p(e^j|x_j)" /></a>
As said earlier, we consider only logistic regression models on discretized data <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;p_\theta(y|e)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;p_\theta(y|e)" title="p_\theta(y|e)" /></a>. Additionnally, it seems like we have to make further assumptions on the nature of the relationship of <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;e^j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;e^j" title="e^j" /></a> to <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;x_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;x_j" title="x_j" /></a>. We chose to use polytomous logistic regressions for continuous <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;X_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;X_j" title="X_j" /></a> and contengency tables for qualitative <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;X_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;X_j" title="X_j" /></a>. This is an arbitrary choice and future versions will include the possibility of plugging your own model.

The first equation becomes:
<a href="https://www.codecogs.com/eqnedit.php?latex=\forall&space;\:&space;x,y,&space;\;&space;p(y|x)&space;=&space;\sum_e&space;p_\theta(y|e)&space;\prod_{j=1}^d&space;p_{\alpha_j}(e^j|x_j)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\forall&space;\:&space;x,y,&space;\;&space;p(y|x)&space;=&space;\sum_e&space;p_\theta(y|e)&space;\prod_{j=1}^d&space;p_{\alpha_j}(e^j|x_j)" title="\forall \: x,y, \; p(y|x) = \sum_e p_\theta(y|e) \prod_{j=1}^d p_{\alpha_j}(e^j|x_j)" /></a>


## The SEM algorithm

It is still hard to optimize over <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;p(y|x;\theta,\alpha)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;p(y|x;\theta,\alpha)" title="p(y|x;\theta,\alpha)" /></a> as the number of candidate discretizations is gigantic as said earlier.

However, calculating <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;p(y,e|x)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;p(y,e|x)" title="p(y,e|x)" /></a> is easy:
<a href="https://www.codecogs.com/eqnedit.php?latex=\forall&space;\:&space;x,y,&space;\;&space;p(y,e|x)&space;=&space;p_\theta(y|e)&space;\prod_{j=1}^d&space;p_{\alpha_j}(e^j|x_j)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\forall&space;\:&space;x,y,&space;\;&space;p(y,e|x)&space;=&space;p_\theta(y|e)&space;\prod_{j=1}^d&space;p_{\alpha_j}(e^j|x_j)" title="\forall \: x,y, \; p(y,e|x) = p_\theta(y|e) \prod_{j=1}^d p_{\alpha_j}(e^j|x_j)" /></a>

As a consequence, we will draw random candidates <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;e" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;e" title="e" /></a> approximately at the mode of the distribution <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;p(y,\cdot|x)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;p(y,\cdot|x)" title="p(y,\cdot|x)" /></a> using an SEM algorithm (see References section).



## Gibbs sampling

To update, at each random draw, the parameters <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\theta" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\theta" title="\theta" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\alpha" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\alpha" title="\alpha" /></a> and propose a new discretization <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;e" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;e" title="e" /></a>, we use the following equation:
<a href="https://www.codecogs.com/eqnedit.php?latex=p(e^j|x_j,y,e^{\{-j\}})&space;\propto&space;p_\theta(y|e)&space;p_{\alpha_j}(e^j|x_j)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(e^j|x_j,y,e^{\{-j\}})&space;\propto&space;p_\theta(y|e)&space;p_{\alpha_j}(e^j|x_j)" title="p(e^j|x_j,y,e^{\{-j\}}) \propto p_\theta(y|e) p_{\alpha_j}(e^j|x_j)" /></a>
Note that we draw <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;e^j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;e^j" title="e^j" /></a> knowing all other variables, especially <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;e^{-j}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;e^{-j}" title="e^{-j}" /></a> so that we introduced a Gibbs sampler (see References section).




# The `glmdisc` package

## The `glmdisc` class

The documentation is available as a [Github Page](https://adimajo.github.io/glmdisc_python/index.html).

The `glmdisc` class implements the algorithm described in the previous section. Its parameters are described first, then its internals are briefly discussed. We finally focus on its ouptuts.

### Parameters

The number of iterations in the SEM algorithm is controlled through the `iter` parameter. It can be useful to first run the `glmdisc` function with a low (10-50) `iter` parameter so you can have a better idea of how much time your code will run.

The `validation` and `test` boolean parameters control if the provided dataset should be divided into training, validation and/or test sets. The validation set aims at evaluating the quality of the model fit at each iteration while the test set provides the quality measure of the final chosen model.

The `criterion` parameters lets the user choose between standard model selection statistics like `aic` and `bic` and the `gini` index performance measure (proportional to the more traditional AUC measure). Note that if `validation=TRUE`, there is no need to penalize the log-likelihood and `aic` and `bic` become equivalent. On the contrary if `criterion="gini"` and `validation=FALSE` then the algorithm may overfit the training data.

The `m_start` parameter controls the maximum number of categories of <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\mathfrak{q}_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\mathfrak{q}_j" title="\mathfrak{q}_j" /></a> for <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;X_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;X_j" title="X_j" /></a> continuous. The SEM algorithm will start with random <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\mathfrak{q}_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\mathfrak{q}_j" title="\mathfrak{q}_j" /></a> taking values in <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\{1,m_{\text{start}}\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\{1,m_{\text{start}}\}" title="\{1,m_{\text{start}}\}" /></a>. For qualitative features <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;X_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;X_j" title="X_j" /></a>, <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\mathfrak{q}_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\mathfrak{q}_j" title="\mathfrak{q}_j" /></a> is initialized with as many values as <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;X_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;X_j" title="X_j" /></a> so that `m_start` has no effect.

Empirical studies show that with a reasonably small training dataset (< 10,000 rows) and a small `m_start` parameter (< 20), approximately 500 to 1500 iterations are largely sufficient to obtain a satisfactory model <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;p_\theta(y|e)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;p_\theta(y|q(x))" title="p_\theta(y|q(x))" /></a>.

### The `fit` function

The `fit` function of the `glmdisc` class is used to run the algorithm over the data provided to it. Subsequently, its parameters are: `predictors_cont` and `predictors_qual` which represent respectively the continuous features to be discretized and the categorical features which values are to be regrouped. They must be of type numpy array, filled with numeric and strings respectively. The last parameter is the class `labels`, of type numpy array as well, in binary form (0/1).

### The `best_formula` function

The `best_formula` function prints out in the console: the cut-points found for continuous features, the regroupments made for categorical features' values. It also returns it in a list.

### The `discrete_data` function

The `discrete_data` function returns the discretized / regrouped version of the `predictors_cont` and `predictors_qual` arguments using the best discretization scheme found so far.

### The `discretize` function

The `discretize` function discretizes a new input dataset in the `predictors_cont`, `predictors_qual` format using the best discretization scheme found so far. The result is a numpy array of the size of the original data.

### The `discretize_dummy` function

The `discretize_dummy` function discretizes a new input dataset in the `predictors_cont`, `predictors_qual` format using the best discretization scheme found so far. The result is a dummy (0/1) numpy array  corresponding to the One-Hot Encoding of the result provided by the `discretize` function.

### The `predict` function

The `predict` function discretizes a new input dataset in the `predictors_cont`, `predictors_qual` format using the best discretization scheme found so far through the `discretizeDummy` function and then applies the corresponding best Logistic Regression model <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;p_\theta(y|e)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;p_\theta(y|e)" title="p_\theta(y|e)" /></a> found so far.

### The attributes

All parameters are stored as attributes: `test`, 
`validation`, `criterion`, `iter`, `m_start` as well as:

* `criterion_iter`: list of values of the criterion chosen;
* `best_link`: link function of the best quantization;
* `best_reglog`: logistic regression function of the best quantization;
* `affectations`: list of label encoders for categorical features;
* `best_encoder_emap`: one hot encoder of the best quantization;
* `performance`: value of the chosen criterion for the best quantization;
* `train`: array of row indices for training samples;
* `validate`: array of row indices for validation samples;
* `test_rows`: array of row indices for test samples;

To see the package in action, please refer to [the accompanying Jupyter Notebook](examples/).

- [ ] Do a notebook

## Authors

* [Adrien Ehrhardt](https://adimajo.github.io)
* [Vincent Vandewalle](https://sites.google.com/site/vvandewa/)
* [Philippe Heinrich](http://math.univ-lille1.fr/~heinrich/)
* [Christophe Biernacki](http://math.univ-lille1.fr/~biernack/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This research has been financed by [Crédit Agricole Consumer Finance](https://www.ca-consumerfinance.com/en.html) through a CIFRE PhD.

This research is supported by [Inria Lille - Nord-Europe](https://www.inria.fr/centre/lille) and [Lille University](https://www.univ-lille.fr/en/home/) as part of a PḧD.

## References

Ehrhardt, A. (2019), Formalization and study of statistical problems in Credit Scoring: Reject inference, discretization and pairwise interactions, logistic regression trees ([PhD thesis](https://github.com/adimajo/manuscrit_these)).

Ehrhardt, A., et al. Feature quantization for parsimonious and interpretable predictive models. [arXiv preprint arXiv:1903.08920 (2019)](https://arxiv.org/abs/1903.08920).

Celeux, G., Chauveau, D., Diebolt, J. (1995), On Stochastic Versions of the EM Algorithm. [Research Report] RR-2514, INRIA. 1995. <inria-00074164>

Agresti, A. (2002) **Categorical Data**. Second edition. Wiley.

Ramírez‐Gallego, S., García, S., Mouriño‐Talín, H., Martínez‐Rego, D., Bolón‐Canedo, V., Alonso‐Betanzos, A. and Herrera, F. (2016). Data discretization: taxonomy and big data challenge. *Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery*, 6(1), 5-21.

## Future development: integration of interaction discovery

Very often, predictive features $X$ "interact" with each other with respect to the response feature. This is classical in the context of Credit Scoring or biostatistics (only the simultaneous presence of several features - genes, SNP, etc. is predictive of a disease).

With the growing number of potential predictors and the time required to manually analyze if an interaction should be added or not, there is a strong need for automatic procedures that screen potential interaction variables. This will be the subject of future work.

## Future development: possibility of changing model assumptions

In the third section, we described two fundamental modelling hypotheses that were made:
>- The real probability density function $p(Y|X)$ can be approximated by a logistic regression $p_\theta(Y|E)$ on the discretized data $E$.
>- The nature of the relationship of $\mathfrak{q}_j$ to $X_j$ is:
>- A polytomous logistic regression if $X_j$ is continuous;
>- A contengency table if $X_j$ is qualitative.

These hypotheses are "building blocks" that could be changed at the modeller's will: discretization could optimize other models.

- [ ] To delete when done with

```{r, echo=TRUE, results='asis'}
x = matrix(runif(1000), nrow = 1000, ncol = 1)
p = 1/(1+exp(-3*x^5))
y = rbinom(1000,1,p)
modele_lin <- glm(y ~ x, family = binomial(link="logit"))
pred_lin <- predict(modele_lin,as.data.frame(x),type="response")
pred_lin_logit <- predict(modele_lin,as.data.frame(x))
```

```{r, echo=FALSE}
knitr::kable(head(data.frame(True_prob = p,Pred_lin = pred_lin)))
```

```{r, echo=TRUE, results='asis'}
x_disc <- factor(cut(x,c(-Inf,0.5,0.7,0.8,0.9,+Inf)),labels = c(1,2,3,4,5))
modele_disc <- glm(y ~ x_disc, family = binomial(link="logit"))
pred_disc <- predict(modele_disc,as.data.frame(x_disc),type="response")
pred_disc_logit <- predict(modele_disc,as.data.frame(x_disc))

```

```{r, echo=FALSE}

knitr::kable(head(data.frame(True_prob = p,Pred_lin = pred_lin,Pred_disc = pred_disc)))
plot(x,3*x^5,main = "Estimated logit transform of p(Y|X)", ylab = "p(Y|X) under different models")
lines(x,pred_lin_logit,type="p",col="red")
lines(x,pred_disc_logit,type="p",col="blue")

```

```{r, echo=TRUE, results='asis'}
x_disc_bad_idea <- factor(cut(x,c(-Inf,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,+Inf)),labels = c(1,2,3,4,5,6,7,8,9,10))
```

```{r, echo=FALSE, results='asis'}
liste_coef <- list()

for (k in 1:10) {
     x_part <- factor(x_disc_bad_idea[((k-1)*nrow(x)/10 +1) : (k/10*nrow(x))])
     y_part <- y[((k-1)*length(y)/10 +1) : (k/10*length(y))]
     modele_part <- glm(y_part ~ x_part, family=binomial(link = "logit"))
     liste_coef[[k]] <- (modele_part$coefficients)
}

estim_coef <- matrix(NA, nrow = nlevels(x_disc_bad_idea), ncol = 10)

for (i in 1:nlevels(x_disc_bad_idea)) {
     estim_coef[i,] <- unlist(lapply(liste_coef,function(batch) batch[paste0("x_part",levels(factor(x_disc_bad_idea))[i])]))
}

stats_coef <- matrix(NA, nrow = nlevels(x_disc_bad_idea), ncol = 3)

for (i in 1:nlevels(x_disc_bad_idea)) {
     stats_coef[i,1] <- mean(estim_coef[i,], na.rm = TRUE)
     stats_coef[i,2] <- sd(estim_coef[i,], na.rm = TRUE)
     stats_coef[i,3] <- sum(is.na(estim_coef[i,]))
}

stats_coef <- stats_coef[-1,]
row.names(stats_coef) <- levels(x_disc_bad_idea)[2:nlevels(x_disc_bad_idea)]

plot (row.names(stats_coef), stats_coef[,1],ylab="Estimated coefficient",xlab="Factor value of x", ylim = c(-1,8))
segments(as.numeric(row.names(stats_coef)), stats_coef[,1]-stats_coef[,2],as.numeric(row.names(stats_coef)),stats_coef[,1]+stats_coef[,2])
lines(row.names(stats_coef),rep(0,length(row.names(stats_coef))),col="red")
```

### Results

First we simulate a "true" underlying discrete model:
```{r, echo=TRUE, results='asis'}
x = matrix(runif(300), nrow = 100, ncol = 3)
cuts = seq(0,1,length.out= 4)
xd = apply(x,2, function(col) as.numeric(cut(col,cuts)))
theta = t(matrix(c(0,0,0,2,2,2,-2,-2,-2),ncol=3,nrow=3))
log_odd = rowSums(t(sapply(seq_along(xd[,1]), function(row_id) sapply(seq_along(xd[row_id,]),
function(element) theta[xd[row_id,element],element]))))
y = rbinom(100,1,1/(1+exp(-log_odd)))
```

The `glmdisc` function will try to "recover" the hidden true discretization `xd` when provided only with `x` and `y`:
```{r, echo=TRUE,warning=FALSE, message=FALSE, results='hide',eval=FALSE}
library(glmdisc)
discretization <- glmdisc(x,y,iter=50,m_start=5,test=FALSE,validation=FALSE,criterion="aic",interact=FALSE)
```

```{r, echo=FALSE,warning=FALSE, message=FALSE, results='hide',eval=TRUE}
library(glmdisc)
discretization <- glmdisc(x,y,iter=50,m_start=5,test=FALSE,validation=FALSE,criterion="aic",interact=FALSE)
```

### How well did we do?

To compare the estimated and the true discretization schemes, we can represent them with respect to the input "raw" data `x`:
<!--```{r, echo=TRUE, out.width='.49\\linewidth', fig.width=3, fig.height=3,fig.show='hold'}-->
```{r, echo=FALSE}
plot(x[,1],xd[,1])
plot(discretization@cont.data[,1],discretization@disc.data[,1])
```

## Contribute

You can clone this project using:

```PowerShell
git clone https://github.com/adimajo/glmdisc_python.git
```

You can install all dependencies, including development dependencies, using (note that 
this command requires `pipenv` which can be installed by typing `pip install pipenv`):

```PowerShell
pipenv install -d
```

You can build the documentation by going into the `docs` directory and typing `make html`.

NOTE: you need to have a separate folder named `glmdisc_python_docs` in the same directory as this repository,
as it will build the docs there so as to allow me to push this other directory as a separate `gh-pages` branch.

You can run the tests by typing `coverage run -m pytest`, which relies on packages 
[coverage](https://coverage.readthedocs.io/en/coverage-5.2/) and [pytest](https://docs.pytest.org/en/latest/).

To run the tests in different environments (one for each version of Python), install `pyenv` (see [the instructions here](https://github.com/pyenv/pyenv)),
install all versions you want to test (see [tox.ini](tox.ini)), e.g. with `pyenv install 3.7.0` and run 
`pipenv run pyenv local 3.7.0 [...]` (and all other versions) followed by `pipenv run tox`.
 