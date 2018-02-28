---
title: "The `glmdisc` package: discretization at its finest"
author: "Adrien Ehrhardt"
date: "`r Sys.Date()`"
output: rmarkdown::html_vignette
vignette: >
  %\VignetteIndexEntry{`glmdisc} package}
  %\VignetteEngine{knitr::rmarkdown}
  %\VignetteEncoding{UTF-8}
---

```{r setup, include = FALSE}
knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>",
  fig.width = 7,
  fig.align = "center",
  fig.height = 4
)
```

# Context

This research has been financed by Crédit Agricole Consumer Finance (CA CF), subsidiary of the Crédit Agricole Group which provides all kinds of banking and insurance services. CA CF specializes in consumer loans. It is a joint work at [Inria Nord-Europe](https://www.inria.fr/en/centre/lille) between Adrien Ehrhardt (CA CF, Inria), Christophe Biernacki (Inria, Lille University), Vincent Vandewalle (Inria, Lille University) and Philippe Heinrich (Lille University).

In order to accept / reject loan applications more efficiently (both quicker and to select better applicants), most financial institutions resort to Credit Scoring: given the applicant's characteristics he/she is given a Credit Score, which has been statistically designed using previously accepted applicants, and which partly decides whether the financial institution will grant the loan or not.

The current methodology for building a Credit Score in most financial institutions is based on logistic regression for several (mostly practical) reasons: it generally gives satisfactory discriminant results, it is reasonably well explainable (contrary to a random forest for example), it is robust to missing data (in particular not financed clients) that follow a MAR missingness mechanism.

## Example

In practice, the statistical modeler has historical data about each customer's characteristics. For obvious reasons, only data available at the time of inquiry must be used to build a future application scorecard. Those data often take the form of a well-structured table with one line per client alongside their performance (did they pay back their loan or not?) as can be seen in the following table:

```{r, echo=FALSE, results='asis'}
knitr::kable(data.frame(Job=c("Craftsman","Technician","Executive","Office employee"),Habitation = c("Owner","Renter","Starter","By family"),Time_in_job = c(10,20,5,2), Children = c(0,1,2,3), Family_status=  c("Divorced","Widower","Single","Married"),Default = c("No","No","Yes","No")))
```

## Notations

In the rest of the vignette, the random vector $X=(X^j)_1^d$ will designate the predictive features, i.e. the characteristics of a client. The random variable $Y \in \{0,1\}$ will designate the label, i.e. if the client has defaulted ($Y=1$) or not ($Y=0$).

We are provided with an i.i.d. sample $(\bar{x},\bar{y}) = (x_i,y_i)_1^n$ consisting in $n$ observations of $X$ and $Y$.

## Logistic regression

The logistic regression model assumes the following relation between $X$ (supposed continuous here) and $Y$:
$$\ln \left( \frac{p_\theta(Y=1|x)}{p_\theta(Y=0|x)} \right) = \theta_0 + \sum_{j=1}^d \theta_j*x^j  $$
where $\theta = (\theta_j)_0^d$ are estimated using $(\bar{x},\bar{y})$.

Clearly, the model assumes linearity of the logit transform of the response $Y$ with respect to $X$.

## Common problems with logistic regression on "raw" data

Fitting a logistic regression model on "raw" data presents several problems, among which some are tackled here.

### Feature selection

First, among all collected information on individuals, some are irrelevant for predicting $Y$. Their coefficient $\theta_j$ should subsequently be $0$ which might (eventually) be the case asymptotically (i.e. $n \rightarrow \infty$).

Second, some collected information are highly correlated and affect each other's coefficient estimation.

As a consequence, data scientists often perform feature selection before training a machine learning algorithm such as logistic regression.

There already exists methods and packages to perform feature selection, see for example the `caret` package.

`glmdisc` is not a feature selection tool but acts as such as a side-effect as we will see in the next part.

### Linearity 

When provided with continuous features, the logistic regression model assumes linearity of the logit transform of the response $Y$ with respect to $X$. This might not be the case at all.

For example, we can simulate a logistic model with an arbitrary power of $X$ and then try to fit a linear logistic model:

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

Of course, providing the `glm` function with a `formula` object containing $X^5$ would solve the problem. This can't be done in practice for two reasons: first, it is too time-consuming to examine all features and candidate polynomials; second, we lose the interpretability of the logistic decision function which was of primary interest.

Consequently, we wish to discretize the input variable $X$ into a categorical feature which will "minimize" the error with respect to the "true" underlying relation:

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

### Too many values per categorical feature

When provided with categorical features, the logistic regression model fits a coefficient for all its values (except one which is taken as a reference). A common problem arises when there are too many values as each value will be taken by a small number of observations $x_i^j$ which makes the estimation of a logistic regression coefficient unstable:

```{r, echo=TRUE, results='asis'}
x_disc_bad_idea <- factor(cut(x,c(-Inf,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,+Inf)),labels = c(1,2,3,4,5,6,7,8,9,10))
```

If we divide the training set in 10 and estimate the variance of each coefficient, we get:

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

All intervals crossing $0$ are non-significant! We should group factor values to get a stable estimation and (hopefully) significant coefficient values.


# Discretization and grouping: theoretical background

## Notations

Let $E=(E^j)_1^d$ be the latent discretized transform of $X$, i.e. taking values in $\{0,\ldots,m_j\}$ where the number of values of each covariate $m_j$ is also latent.

The fitted logistic regression model is now:
$$\ln \left( \frac{p_\theta(Y=1|e)}{p_\theta(Y=0|e)} \right) = \theta_0 + \sum_{j=1}^d \sum_{k=1}^{m_j} \theta^j_k*\mathbb{1}_{e^j=k}$$
Clearly, the number of parameters has grown which allows for flexible approximation of the true underlying model $p(Y|E)$.

## Best discretization?

Our goal is to obtain the model $p_\theta(Y|e)$ with best predictive power. As $E$ and $\theta$ are both optimized, a formal goodness-of-fit criterion could be:
$$ (\hat{\theta},\hat{\bar{e}}) = \arg \max_{\theta,\bar{e}} \text{AIC}(p_\theta(\bar{y}|\bar{e})) $$
where AIC stands for Akaike Information Criterion.

## Combinatorics

The problem seems well-posed: if we were able to generate all discretization schemes transforming $X$ to $E$, learn $p_\theta(y|e)$ for each of them and compare their AIC values, the problem would be solved.

Unfortunately, there are way too many candidates to follow this procedure. Suppose we want to construct $k$ intervals of $E^j$ given $n$ distinct $(x^j_i)_1^n$. There is $n \choose k$ models. The true value of $k$ is unknown, so it must be looped over. Finally, as logistic regression is a multivariate model, the discretization of $E^j$ can influence the discretization of $E^k$, $k \neq j$. 

As a consequence, existing approaches to discretization (in particular discretization of continuous attributes) rely on strong assumptions to simplify the search of good candidates as can be seen in the review of Ramírez‐Gallego, S. et al. (2016) - see References section.

# Discretization and grouping: estimation

## Likelihood estimation

$E$ can be introduced in $p(Y|X)$:
$$\forall \: x,y, \; p(y|x) = \sum_e p(y|x,e)p(e|x)$$

First, we assume that all information about $Y$ in $X$ is already contained in $E$ so that:
$$\forall \: x,y,e, \; p(y|x,e)=p(y|e)$$
Second, we assume the conditional independence of $E^j$ given $X^j$, i.e. knowing $X^j$, the discretization $E^j$ is independent of the other features $X^k$ and $E^k$ for all $k \neq j$:
$$\forall \:x, k\neq j, \; E^j | x^j \perp E^k | x^k$$
The first equation becomes:
$$\forall \: x,y, \; p(y|x) = \sum_e p(y|e) \prod_{j=1}^d p(e^j|x^j)$$
As said earlier, we consider only logistic regression models on discretized data $p_\theta(y|e)$. Additionnally, it seems like we have to make further assumptions on the nature of the relationship of $e^j$ to $x^j$. We chose to use polytomous logistic regressions for continuous $X^j$ and contengency tables for qualitative $X^j$. This is an arbitrary choice and future versions will include the possibility of plugging your own model.

The first equation becomes:
$$\forall \: x,y, \; p(y|x) = \sum_e p_\theta(y|e) \prod_{j=1}^d p_{\alpha_j}(e^j|x^j)$$

## The SEM algorithm

It is still hard to optimize over $p(y|x;\theta,\alpha)$ as the number of candidate discretizations is gigantic as said earlier.

However, calculating $p(y,e|x)$ is easy:
$$\forall \: x,y, \; p(y,e|x) = p_\theta(y|e) \prod_{j=1}^d p_{\alpha_j}(e^j|x^j)$$

As a consequence, we will draw random candidates $e$ approximately at the mode of the distribution $p(y,\cdot|x)$ using an SEM algorithm (see References section).

## Gibbs sampling

To update, at each random draw, the parameters $\theta$ and $\alpha$ and propose a new discretization $e$, we use the following equation:
$$p(e^j|x^j,y,e^{\{-j\}}) \propto p_\theta(y|e) p_{\alpha_j}(e^j|x^j)$$
Note that we draw $e^j$ knowing all other variables, especially $e^{-j}$ so that we introduced a Gibbs sampler (see References section).

# The `glmdisc` package

## The `glmdisc` function

The `glmdisc` function implements the algorithm discribed in the previous section. Its parameters are described first, then its internals are briefly discussed. We finally focus on its ouptuts.

### Parameters

The number of iterations in the SEM algorithm is controlled through the `iter` parameter. It can be useful to first run the `glmdisc` function with a low (10-50) `iter` parameter so you can have a better idea of how much time your code will run.

The `validation` and `test` boolean parameters control if the provided dataset should be divided into training, validation and/or test sets. The validation set aims at evaluating the quality of the model fit at each iteration while the test set provides the quality measure of the final chosen model.

The `criterion` parameters lets the user choose between standard model selection statistics like `aic` and `bic` and the `gini` index performance measure (proportional to the more traditional AUC measure). Note that if `validation=TRUE`, there is no need to penalize the log-likelihood and `aic` and `bic` become equivalent. On the contrary if `criterion="gini"` and `validation=FALSE` then the algorithm may overfit the training data.

The `m_start` parameter controls the maximum number of categories of $E^j$ for $X^j$ continuous. The SEM algorithm will start with random $E^j$ taking values in $\{1,m_{\text{start}}\}$. For qualitative features $X^j$, $E^j$ is initialized with as many values as $X^j$ so that `m_start` has no effect.

The `reg_type` parameter controls the model used to fit $E^j$ to continuous $X^j$. The default behavior (`reg_type=poly`) uses `multinom` from the `nnet` package which is a polytomous logistic regression (see References section). The other `reg_type` value is `polr` from the `MASS` package which is an ordered logistic regression (simpler to estimate but less flexible as it fits way fewer parameters).

Empirical studies show that with a reasonably small training dataset ($\leq 100 000$ rows) and a small `m_start` parameter ($\leq 20$), approximately $500$ to $1500$ iterations are largely sufficient to obtain a satisfactory model $p_\theta(y|e)$.

### Internals

First, the discretized version $E$ of $X$ is initialized at random using the user-provided maximum number of values for quantitative values through the `m_start` parameter.

Then we iterate `times`:
First, the model $p_\theta(y|e)$, where $e$ denotes the current value of $E$, is adjusted. Then, for each feature $X^j$, the model $p_{\alpha_j}(e|x)$ is adjusted (depending on the type of $X^j$ it can be a polytomous logistic regression or a contengency table). Finally, we draw a new candidate $e^j$ with probability $p_\theta(y|e)p_{\alpha_j}(e^j|x^j)$.

The performance metric chosen through the `criterion` parameter determines the best candidate $e$.

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

### The `parameters` slot

The `parameters` slot refers back to the user-provided parameters given to the `glmdisc` function. Consequently, users can compare results of different `glmdisc` with respect to their parameters.

```{r, echo=TRUE}
discretization@parameters
```

### The `best.disc` slot

The `best.disc` slot contains all that is needed to reproduce the best discretization scheme obtained by `glmdisc` function: the first element is the best logistic regression model obtained $p_\theta(y|e)$; the second element is its associated "link" function, i.e. either the `multinom` or `polr` functions fit to each continuous features and the contengency tables fit to each quantitative features.
```{r, echo=TRUE}
discretization@best.disc[[1]]

# The first link function is:
discretization@best.disc[[2]][[1]]
```

### The `performance` slot

The `performance` slot lists both the achieved performance defined by the parameters `criterion`, `test` and `validation` and this performance metric over all the iterations.

`discretization` was fit with no test nor validation and we used the AIC criterion. The best AIC achieved is therefore:
```{r, echo=TRUE}
discretization@performance[[1]]
```

The first 5 AIC values obtained on the training set over the iterations are:
```{r, echo=TRUE}
discretization@performance[[2]][1:5]
```

### The `disc.data` slot

The `disc.data` slot contains the discretized data obtained by applying the best discretization scheme found on the test set if `test=TRUE` or the validation set if `validation=TRUE` or the training set.

In our example, `discretization` has `validation=FALSE` and `test=FALSE` so that we get the discretized training set:
```{r, echo=TRUE, eval=FALSE}
discretization@disc.data
```

```{r, echo=FALSE}
knitr::kable(head(discretization@disc.data))
```

### The `cont.data` slot

The `cont.data` slot contains the original "continuous" counterpart of the `disc.data` slot.

In our example, `discretization` has `validation=FALSE` and `test=FALSE` so that we get the "raw" training set:
```{r, echo=TRUE,eval=FALSE}
discretization@cont.data
```

```{r, echo=FALSE}
knitr::kable(head(discretization@cont.data))
```

### How well did we do?

To compare the estimated and the true discretization schemes, we can represent them with respect to the input "raw" data `x`:
<!--```{r, echo=TRUE, out.width='.49\\linewidth', fig.width=3, fig.height=3,fig.show='hold'}-->
```{r, echo=FALSE}
plot(x[,1],xd[,1])
plot(discretization@cont.data[,1],discretization@disc.data[,1])
```

## "Classical" methods

A few classical R methods have been adapted to the `glmdisc` S4 object.

The `plot` function is the standard `plot.glm` method applied to the best discretization scheme found by the `glmdisc` function:
```{r, echo=TRUE}
# plot(discretization)
```

The `print` function is the standard `print.summary.glm` method applied to the best discretization scheme found by the `glmdisc` function:
```{r, echo=TRUE}
print(discretization)
```

Its S4 equivalent `show` is also available: 
```{r, echo=TRUE}
show(discretization)
```

## The `discretize` method

The `discretize` method allows the user to discretize a new "raw" input set by applying the best discretization scheme found by the `glmdisc` function in a provided `glmdisc` object.
```{r, echo=TRUE, results='asis'}
x_new <- discretize(discretization,x)
```

```{r, echo=FALSE, warning=FALSE}
knitr::kable(head(x_new))
```
## The `predict` method

The `discretize` method allows the user to predict the response of a new "raw" input set by first discretizing it using the previously described `discretize` method and a previously trained `glmdisc` object. It then predicts using the standard `predict.glm` method and the best discretization scheme found by the `glmdisc` function.
```{r, echo=TRUE}
pred_new <- predict(discretization,x)
```

```{r, echo=FALSE, warning=FALSE}
knitr::kable(head(pred_new))
```

# Future developments

## Enhancing the `discretization` package

The `discretization` package available from CRAN implements some existing supervised univariate discretization methods (applicable only to continuous inputs) : ChiMerge (`chiM`), Chi2 (`chi2`), Extended Chi2 (`extendChi2`), Modified Chi2 (`modChi2`), MDLP (`mdlp`), CAIM, CACC and AMEVA (`disc.Topdown`).

To compare the result of such methods to the one we propose, the package `discretization` will be enhanced to support missing values (by putting them in a single class for example) and quantitative features (through a Chi2 grouping method for example).

## Integration of interaction discovery

Very often, predictive features $X$ "interact" with each other with respect to the response feature. This is classical in the context of Credit Scoring or biostatistics (only the simultaneous presence of several features - genes, SNP, etc. is predictive of a disease).

With the growing number of potential predictors and the time required to manually analyze if an interaction should be added or not, there is a strong need for automatic procedures that screen potential interaction variables. This will be the subject of future work.

## Possibility of changing model assumptions

In the third section, we described two fundamental modelling hypotheses that were made:
>- The real probability density function $p(Y|X)$ can be approximated by a logistic regression $p_\theta(Y|E)$ on the discretized data $E$.
>- The nature of the relationship of $E^j$ to $X^j$ is:
>- A polytomous logistic regression if $X^j$ is continuous;
>- A contengency table if $X^j$ is qualitative.

These hypotheses are "building blocks" that could be changed at the modeller's will: discretization could optimize other models.

## Make it faster

The current implementation is heavily based on R code. A lot of logistic regression are fit, either through `multinom`, `polr` or `glm` calls, which can probably be sped up by an intelligent initialization of their parameters and a limitation or their number of iterations.

We also loop of qualitative features and their values, which is known to be slow in R. A C++ version of a Gibbs sampler (which is used here) described by Hadley Wickham in [Advanced R](http://adv-r.had.co.nz/Rcpp.html#rcpp-case-studies) shows a potential speed up by 40.

# References

Celeux, G., Chauveau, D., Diebolt, J. (1995), On Stochastic Versions of the EM Algorithm. [Research Report] RR-2514, INRIA. 1995. <inria-00074164>

Agresti, A. (2002) **Categorical Data**. Second edition. Wiley.

Ramírez‐Gallego, S., García, S., Mouriño‐Talín, H., Martínez‐Rego, D., Bolón‐Canedo, V., Alonso‐Betanzos, A. and Herrera, F. (2016). Data discretization: taxonomy and big data challenge. *Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery*, 6(1), 5-21.
