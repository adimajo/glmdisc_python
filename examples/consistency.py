import numpy as np
import glmdisc
from tensorflow.keras.optimizers import RMSprop


if __name__ == "__main__":

    n = 100000
    d = 2
    theta = np.array([[0] * d] * 3)
    theta[1, :] = 3
    theta[2, :] = -3

    for _ in range(100):
        x, y, theta = glmdisc.Glmdisc.generate_data(n, d, theta)
        model = glmdisc.Glmdisc(algorithm="NN", validation=False, test=False, m_start=3, criterion="aic")
        optim = RMSprop(lr=0.1, rho=0.95, epsilon=None, decay=0.0)
        model.fit(predictors_cont=x, predictors_qual=None, labels=y, iter=20, plot=True, optimizer=optim)
        best_cutpoints = model.best_formula()
