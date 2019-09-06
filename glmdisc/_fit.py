#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 16:16:40 2019

@author: adrien
"""


def fit(self, predictors_cont, predictors_qual, labels):

    """Fits the glmdisc object.

    Keyword arguments:
    predictors_cont -- Continuous predictors to be discretized in a numpy
                        "numeric" array. Can be provided either here or with
                        the __init__ method.
    predictors_qual -- Categorical features which levels are to be merged
                        (also in a numpy "string" array). Can be provided
                        either here or with the __init__ method.
    labels          -- Boolean (0/1) labels of the observations. Must be of
                        the same length as predictors_qual and predictors_cont
                        (numpy "numeric" array).
    """

    # Tester la présence de labels
    if not type(labels) is np.ndarray:
        raise ValueError('glmdisc only supports numpy.ndarray inputs')

    # Tester la présence d'au moins qual ou cont
    if predictors_cont is None and predictors_qual is None:
        raise ValueError(('You must provide either qualitative or quantitative'
                         'features'))

    # Tester la présence de prédicteurs continus
    if predictors_cont is not None:
        # Tester la même longueur que labels
        if predictors_cont.shape[0] != labels.shape[0]:
            raise ValueError('Predictors and labels must be of same size')

    # Tester la présence de prédicteurs catégoriels
    if predictors_qual is not None:
        # Tester la même longueur que labels
        if predictors_qual.shape[0] != labels.shape[0]:
            raise ValueError('Predictors and labels must be of same size')

    self.predictors_cont = predictors_cont
    self.predictors_qual = predictors_qual
    self.labels = labels

    # Calcul des variables locales utilisées dans la suite

    # Calculate shape of predictors (re-used multiple times)
    n = self.labels.shape[0]
    try:
        d1 = self.predictors_cont.shape[1]
    except AttributeError:
        d1 = 0

    try:
        d2 = self.predictors_qual.shape[1]
    except AttributeError:
        d2 = 0

    # Gérer les manquants des variables continues, dans un premier temps
    # comme une modalité à part
    continu_complete_case = np.invert(np.isnan(self.predictors_cont))
    sum_continu_complete_case = np.zeros((n, d1))

    for j in range(d1):
        sum_continu_complete_case[0, j] = continu_complete_case[0, j] * 1
        for l in range(1, n):
            sum_continu_complete_case[l, j] = \
                sum_continu_complete_case[l - 1, j] + \
                continu_complete_case[l, j] * 1

    # Initialization for following the performance of the discretization
    current_best = 0

    # Initial random "discretization"
    self.affectations = [None] * (d1 + d2)
    edisc = np.random.choice(list(range(self.m_start)), size=(n, d1 + d2))

    for j in range(d1):
        edisc[np.invert(continu_complete_case[:, j]), j] = self.m_start

    predictors_trans = np.zeros((n, d2))

    for j in range(d2):
        self.affectations[j + d1] = sk.preprocessing.LabelEncoder().fit(
                self.predictors_qual[:, j])
        if (self.m_start > stats.describe(sk.preprocessing.LabelEncoder().fit(
                self.predictors_qual[:, j]).transform(
                        self.predictors_qual[:, j])).minmax[1] + 1):
            edisc[:, j + d1] = np.random.choice(list(range(
                    stats.describe(sk.preprocessing.LabelEncoder().fit(
                            self.predictors_qual[:, j]).transform(
                                    self.predictors_qual[:, j])).minmax[1])),
                                        size=n)
        else:
            edisc[:, j + d1] = np.random.choice(list(range(self.m_start)),
                                                size=n)

        predictors_trans[:, j] = (self.affectations[j + d1].transform(
                                    self.predictors_qual[:, j])).astype(int)

    emap = np.ndarray.copy(edisc)

    model_edisc = sk.linear_model.LogisticRegression(solver='liblinear',
                                                     C=1e40,
                                                     tol=0.001,
                                                     max_iter=25,
                                                     warm_start=False)

    model_emap = sk.linear_model.LogisticRegression(solver='liblinear',
                                                    C=1e40,
                                                    tol=0.001,
                                                    max_iter=25,
                                                    warm_start=False)

    current_encoder_edisc = sk.preprocessing.OneHotEncoder()

    current_encoder_emap = sk.preprocessing.OneHotEncoder()

    # Initialisation link et m
    link = [None] * (d1 + d2)
    m = [None] * (d1 + d2)

    for j in range(d1):
        link[j] = sk.linear_model.LogisticRegression(C=1e40,
                                                     multi_class='multinomial',
                                                     solver='newton-cg',
                                                     max_iter=25,
                                                     tol=0.001,
                                                     warm_start=False)

    # Random splitting
    if self.validation and self.test:
        train, validate, test_rows = np.split(np.random.choice(n,
                                                               n,
                                                               replace=False),
                                              [int(.6*n), int(.8*n)])
    elif self.validation:
        train, validate = np.split(np.random.choice(n, n, replace=False),
                                   int(.6*n))
    elif self.test:
        train, test_rows = np.split(np.random.choice(n, n, replace=False),
                                    int(.6*n))
    else:
        train = np.random.choice(n, n, replace=False)

    # Itérations MCMC
    for i in range(self.iter):

        # Recalcul des matrices disjonctives
        current_encoder_edisc.fit(X=edisc.astype(str))
        current_encoder_emap.fit(X=emap.astype(str))

        # Apprentissage p(y|e) et p(y|emap)
        try:
            model_edisc.fit(X=current_encoder_edisc.transform(
                    edisc[train, :].astype(str)),
                    y=self.labels[train])
        except ValueError:
            model_edisc.fit(X=current_encoder_edisc.transform(
                    edisc[train, :].astype(str)),
                    y=self.labels[train])

        try:
            model_emap.fit(X=current_encoder_emap.transform(
                    emap[train, :].astype(str)),
                    y=self.labels[train])
        except ValueError:
            model_emap.fit(X=current_encoder_emap.transform(
                    emap[train, :].astype(str)),
                    y=self.labels[train])

        # Calcul du critère
        if self.criterion in ['aic', 'bic']:
            loglik = -sk.metrics.log_loss(self.labels,
                                          model_emap.predict_proba(
                                            X=current_encoder_emap.transform(
                                                  emap[train, :].astype(str))),
                                          normalize=False)
            if self.validation:
                self.criterion_iter.append(loglik)

        if self.criterion == 'aic' and not self.validation:
            self.criterion_iter.append(-(2 * model_emap.coef_.shape[1]
                                       - 2 * loglik))

        if self.criterion == 'bic' and not self.validation:
            self.criterion_iter.append(-(log(n) * model_emap.coef_.shape[1]
                                         - 2 * loglik))

        if self.criterion == 'gini' and self.validation:
            self.criterion_iter.append(sk.metrics.roc_auc_score(
                    self.labels[validate], model_emap.predict_proba(
                            X=current_encoder_emap.transform(
                                    emap[validate, :].astype(str)))))

        if self.criterion == 'gini' and not self.validation:
            self.criterion_iter.append(sk.metrics.roc_auc_score(
                    self.labels[train], model_emap.predict_proba(
                            X=current_encoder_emap.transform(
                                    emap[train, :].astype(str)))))

        # Mise à jour éventuelle du meilleur critère
        if (self.criterion_iter[i] <= self.criterion_iter[current_best]):

            # Update current best logistic regression
            self.best_reglog = model_emap
            self.best_link = link
            current_best = i
            self.best_encoder_emap = current_encoder_emap

        for j in range(d1 + d2):
            m[j] = np.unique(edisc[:, j])

        # On construit la base disjonctive nécessaire au modèle de régression
        # logistique
        base_disjonctive = current_encoder_edisc.transform(
                X=edisc[train, :].astype(str)).toarray()

        # On boucle sur les variables pour le tirage de e^j | reste
        for j in np.random.permutation(d1 + d2):
            # On commence par les quantitatives
            if (j < d1):
                # On apprend e^j | x^j
                link[j].fit(y=edisc[train, :]
                            [continu_complete_case[train, j], j],
                            X=predictors_cont[train, :]
                            [continu_complete_case[train, j], j].reshape(-1, 1))

                y_p = np.zeros((n, len(m[j])))

                # On calcule y | e^{-j} , e^j
                for k in range(len(m[j])):
                    modalites = np.zeros((n, len(m[j])))
                    modalites[:, k] = np.ones((n, ))
                    y_p[:, k] = model_edisc.predict_proba(np.column_stack(
                            (base_disjonctive[:, 0:(sum(list(map(len, m[0:j]))))],
                                modalites,
                                base_disjonctive[:, (sum(list(map(len,
                                                         m[0:(j + 1)])))):(sum(list(map(len, m))))]))
                            )[:, 1] * (2 * np.ravel(self.labels) - 1) - np.ravel(self.labels) + 1

                # On calcule e^j | x^j sur tout le monde
                t = link[j].predict_proba(
                        self.predictors_cont[(continu_complete_case[:,j]), j].reshape(-1, 1))

                # On met à jour emap^j
                emap[(continu_complete_case[:, j]), j] = np.argmax(t, axis=1)
                emap[np.invert(continu_complete_case[:, j]), j] = m[j][-1]

                # On calcule e^j | reste
                if (np.invert(continu_complete_case[:, j]).sum() == 0):
                    t = t * y_p
                else:
                    t = t[:, 0:(len(m[j])-1)] * y_p[continu_complete_case[:, j], 0:(len(m[j]) - 1)]

                t = t / (t.sum(axis=1)[:, None])

                # On met à jour e^j
                edisc[continu_complete_case[:, j], j] = vectorized_multinouilli(t, m[j])
                edisc[np.invert(continu_complete_case[:, j]), j] = max(m[j])

            # Variables qualitatives
            else:
                # On fait le tableau de contingence e^j | x^j
                link[j] = Counter([tuple(element) for element in np.column_stack((predictors_trans[train, j - d1], edisc[train, j]))])

                y_p = np.zeros((n, len(m[j])))

                # On calcule y | e^{-j} , e^j
                for k in range(len(m[j])):
                    modalites = np.zeros((n, len(m[j])))
                    modalites[:, k] = np.ones((n, ))

                    y_p[:, k] = model_edisc.predict_proba(np.column_stack(
                            (base_disjonctive[:, 0:(sum(list(map(len, m[0:j]))))],
                                              modalites,
                                              base_disjonctive[:,(sum(list(map(len, m[0:(j + 1)])))):(sum(list(map(len, m))))])))[:, 1] * (2 * np.ravel(self.labels) - 1) - np.ravel(self.labels) + 1

                t = np.zeros((n, int(len(m[j]))))

                # On calcule e^j | x^j sur tout le monde
                for l in range(n):
                    for k in range(int(len(m[j]))):
                        t[l, k] = link[j][(predictors_trans[l, j - d1], k)] / n

                # On met à jour emap^j
                emap[:, j] = np.argmax(t, axis=1)

                # On calcule e^j | reste
                t = t * y_p
                t = t / (t.sum(axis=1)[:, None])

                edisc[:, j] = vectorized_multinouilli(t, m[j])

        # On regarde si des modalités sont présentes dans validation
        # et pas dans train

    # Fin des itérations MCMC

    # Meilleur(s) modèle(s) et équation de régression logistique

    # Evaluation de la performance
