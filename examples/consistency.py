import numpy as np
import glmdisc


if __name__ == "__main__":

    n = 10000
    d = 2
    theta = np.array([[0] * d] * 3)
    theta[1, :] = 2
    theta[2, :] = -2

    for _ in range(100):
        x, y, theta = glmdisc.Glmdisc.generate_data(n, d, theta)
        model = glmdisc.Glmdisc(algorithm="NN", validation=False, test=False, m_start=3, criterion="aic")
        model.fit(predictors_cont=x, predictors_qual=None, labels=y, iter=400)
        best_cutpoints = model.best_formula()
