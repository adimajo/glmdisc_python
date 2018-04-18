# -*- coding: utf-8 -*-
"""
    This module is dedicated to preprocessing tasks for logistic regression and post-learning graphical tools.
"""

import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.preprocessing
import sklearn.linear_model
import warnings
import matplotlib.pyplot as plt

from scipy import stats
from collections import Counter
from math import log
from pygam import LogisticGAM
from pygam.utils import generate_X_grid

def vectorized(prob_matrix, items):
    s = prob_matrix.cumsum(axis=1)
    r = np.random.rand(prob_matrix.shape[0]).reshape((-1,1))
    k = (s < r).sum(axis=1)
    return items[k]


class glmdisc:
    
    """
    This class implements a supervised multivariate discretization method, factor levels grouping and interaction discovery for logistic regression.
    """

    def __init__(self,test=True,validation=True,criterion="bic",iter=100,m_start=20):
        
        """Initializes self by checking if its arguments are appropriately specified.
        
        Keyword arguments:
        test            -- Boolean (T/F) specifying if a test set is required. If True, the provided data is split to provide 20% of observations in a test set and the reported performance is the Gini index on test set.
        validation      -- Boolean (T/F) specifying if a validation set is required. If True, the provided data is split to provide 20% of observations in a validation set and the reported performance is the Gini index on the validation set (if no test=False). The quality of the discretization at each step is evaluated using the Gini index on the validation set, so criterion must be set to "gini".
        criterion       -- The criterion to be used to assess the goodness-of-fit of the discretization: "bic" or "aic" if no validation set, else "gini".
        iter            -- Number of MCMC steps to perform. The more the better, but it may be more intelligent to use several MCMCs. Computation time can increase dramatically.
        m_start         -- Number of initial discretization intervals for all variables. If m_start is bigger than the number of factor levels for a given variable in predictors_qual, m_start is set (for this variable only) to this variable's number of factor levels.
        """
        
                ################## Tests des variables d'entrée ##################
    
        # Le critère doit être un des trois de la liste
        if criterion not in ['gini','aic','bic']:
            raise ValueError('Criterion must be one of Gini, Aic, Bic')
            
        # test est bool
        if not type(test) is bool:
            raise ValueError('test must be boolean')
        
        # validation est bool
        if not type(validation) is bool:
            raise ValueError('validation must be boolean')
        
        # iter doit être suffisamment grand
        if iter<=10:
            raise ValueError('iter is too low / negative. Please set 10 < iter < 100 000')
        
        # iter doit être suffisamment petit    
        if iter >= 100000:
            raise ValueError('iter is too high, it will take years to finish! Please set 10 < iter < 100 000')
        
        # m_start doit être pas déconnant
        if not 2 <= m_start <= 50:
            raise ValueError('Please set 2 <= m_start <= 50')
            
        if not(validation) and criterion=='gini':
            warnings.warn('Using Gini index on training set might yield an overfitted model')
            
        if validation and criterion in ['aic','bic']:
            warnings.warn('No need to penalize the log-likelihood when a validation set is used. Using log-likelihood instead.')

        
        self.test = test
        self.validation = validation
        self.criterion = criterion
        self.iter = iter
        self.m_start = m_start
        
        self.criterion_iter = []
        self.best_link = []
        self.best_reglog = 0
        self.affectations = []
        self.best_encoder_emap = []
        

    def fit(self,predictors_cont,predictors_qual,labels):
        
        """Fits the glmdisc object.
        
        Keyword arguments:
        predictors_cont -- Continuous predictors to be discretized in a numpy "numeric" array. Can be provided either here or with the __init__ method.
        predictors_qual -- Categorical features which levels are to be merged (also in a numpy "string" array). Can be provided either here or with the __init__ method.
        labels          -- Boolean (0/1) labels of the observations. Must be of the same length as predictors_qual and predictors_cont (numpy "numeric" array).
        """
        
        # Tester la présence de labels
        if not type(labels) is np.ndarray:
            raise ValueError('glmdisc only supports numpy.ndarray inputs')
            
        # Tester la présence d'au moins qual ou cont
        if predictors_cont is None and predictors_qual is None:
            raise ValueError('You must provide either qualitative or quantitative features')
            
        # Tester la présence de prédicteurs continus
        if not predictors_cont is None:
            ## Tester la même longueur que labels
            if predictors_cont.shape[0] != labels.shape[0]:
                raise ValueError('Predictors and labels must be of same size')
        
        # Tester la présence de prédicteurs catégoriels
        if not predictors_qual is None:
            ## Tester la même longueur que labels
            if predictors_qual.shape[0] != labels.shape[0]:
                raise ValueError('Predictors and labels must be of same size')

        self.predictors_cont = predictors_cont
        self.predictors_qual = predictors_qual
        self.labels = labels

        ################## Calcul des variables locales utilisées dans la suite ##################
    
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
    
    
        # Gérer les manquants des variables continues, dans un premier temps comme une modalité à part
        continu_complete_case = np.invert(np.isnan(self.predictors_cont))
        sum_continu_complete_case = np.zeros((n,d1))
    
        for j in range(d1):
            sum_continu_complete_case[0,j] = continu_complete_case[0,j]*1
            for l in range(1,n):
                sum_continu_complete_case[l,j] = sum_continu_complete_case[l-1,j]+continu_complete_case[l,j]*1
    
        # Initialization for following the performance of the discretization
        current_best = 0
        
        # Initial random "discretization"
        self.affectations = [None]*(d1+d2)
        edisc = np.random.choice(list(range(self.m_start)),size=(n,d1+d2))
        
        for j in range(d1):
            edisc[np.invert(continu_complete_case[:,j]),j] = self.m_start
                  
        predictors_trans = np.zeros((n,d2))
        
        for j in range(d2):
            self.affectations[j+d1] = sk.preprocessing.LabelEncoder().fit(self.predictors_qual[:,j])
            if (self.m_start > stats.describe(sk.preprocessing.LabelEncoder().fit(self.predictors_qual[:,j]).transform(self.predictors_qual[:,j])).minmax[1]+1):
                edisc[:,j+d1] = np.random.choice(list(range(stats.describe(sk.preprocessing.LabelEncoder().fit(self.predictors_qual[:,j]).transform(self.predictors_qual[:,j])).minmax[1])), size = n)
            else:
                edisc[:,j+d1] = np.random.choice(list(range(self.m_start)),size = n)
        
            predictors_trans[:,j] = (self.affectations[j+d1].transform(self.predictors_qual[:,j])).astype(int)
        
        emap = np.ndarray.copy(edisc)
        
        model_edisc = sk.linear_model.LogisticRegression(solver='liblinear',C=1e40,tol=0.001,max_iter=25,warm_start=False)
        
        model_emap = sk.linear_model.LogisticRegression(solver='liblinear',C=1e40,tol=0.001,max_iter=25,warm_start=False)
        
        current_encoder_edisc = sk.preprocessing.OneHotEncoder()
        
        current_encoder_emap = sk.preprocessing.OneHotEncoder()
        
        # Initialisation link et m
        link = [None]*(d1+d2)
        m=[None]*(d1+d2)        
    
        for j in range(d1):
            link[j] = sk.linear_model.LogisticRegression(C=1e40,multi_class='multinomial',solver='newton-cg',max_iter=25,tol=0.001,warm_start=False)
        
        ################## Random splitting ##################
    
        if self.validation and self.test:
            train, validate, test_rows = np.split(np.random.choice(n,n,replace=False), [int(.6*n), int(.8*n)])
        elif self.validation:
            train, validate = np.split(np.random.choice(n,n,replace=False), int(.6*n))
        elif self.test:
            train, test_rows = np.split(np.random.choice(n,n,replace=False), int(.6*n))
        else:
            train = np.random.choice(n,n,replace=False)
    
        ################## Itérations MCMC ##################
    
        for i in range(self.iter):
            
            # Recalcul des matrices disjonctives
            current_encoder_edisc.fit(X=edisc.astype(str))
            current_encoder_emap.fit(X=emap.astype(str))
        
            # Apprentissage p(y|e) et p(y|emap)
            try:
                model_edisc.fit(X = current_encoder_edisc.transform(edisc[train,:].astype(str)), y = self.labels[train])
            except ValueError:
                model_edisc.fit(X = current_encoder_edisc.transform(edisc[train,:].astype(str)), y = self.labels[train])
    
            try:
                model_emap.fit(X = current_encoder_emap.transform(emap[train,:].astype(str)), y = self.labels[train])
            except ValueError:
                model_emap.fit(X = current_encoder_emap.transform(emap[train,:].astype(str)), y = self.labels[train])
    
            # Calcul du critère
            if self.criterion in ['aic','bic']:
                loglik = -sk.metrics.log_loss(self.labels,model_emap.predict_proba(X = current_encoder_emap.transform(emap[train,:].astype(str))), normalize=False)
                if self.validation:
                    self.criterion_iter.append(loglik)
                
            if self.criterion == 'aic' and not self.validation:
                self.criterion_iter.append(-(2*model_emap.coef_.shape[1] -2*loglik))
            
            if self.criterion == 'bic' and not self.validation:
                self.criterion_iter.append(-(log(n)*model_emap.coef_.shape[1] -2*loglik))
                
            if self.criterion == 'gini' and self.validation:
                self.criterion_iter.append(sk.metrics.roc_auc_score(self.labels[validate],model_emap.predict_proba(X = current_encoder_emap.transform(emap[validate,:].astype(str)))))
                
            if self.criterion == 'gini' and not self.validation:
                self.criterion_iter.append(sk.metrics.roc_auc_score(self.labels[train],model_emap.predict_proba(X = current_encoder_emap.transform(emap[train,:].astype(str)))))
                
            
            # Mise à jour éventuelle du meilleur critère
            if (self.criterion_iter[i] <= self.criterion_iter[current_best]):
                
                # Update current best logistic regression
                self.best_reglog = model_emap
                self.best_link = link
                current_best = i
                self.best_encoder_emap = current_encoder_emap
            
            for j in range(d1+d2):
                m[j] = np.unique(edisc[:,j])
            
            # On construit la base disjonctive nécessaire au modèle de régression logistique
            base_disjonctive = current_encoder_edisc.transform(X=edisc[train,:].astype(str)).toarray()
    
            # On boucle sur les variables pour le tirage de e^j | reste
            for j in np.random.permutation(d1+d2):
                # On commence par les quantitatives
                if (j<d1):
                    # On apprend e^j | x^j
                    link[j].fit(y=edisc[train,:][continu_complete_case[train,j],j],X=predictors_cont[train,:][continu_complete_case[train,j],j].reshape(-1,1))
            
                    y_p = np.zeros((n,len(m[j])))
                    
                    # On calcule y | e^{-j} , e^j
                    for k in range(len(m[j])):
                        modalites = np.zeros((n,len(m[j])))
                        modalites[:,k] = np.ones((n,))
                        y_p[:,k] = model_edisc.predict_proba(np.column_stack((base_disjonctive[:,0:(sum(list(map(len,m[0:j]))))],modalites,base_disjonctive[:,(sum(list(map(len,m[0:(j+1)])))):(sum(list(map(len,m))))])))[:,1]*(2*np.ravel(self.labels)-1)-np.ravel(self.labels)+1        
                
                    # On calcule e^j | x^j sur tout le monde
                    t = link[j].predict_proba(self.predictors_cont[(continu_complete_case[:,j]),j].reshape(-1,1))
                    
                    # On met à jour emap^j
                    emap[(continu_complete_case[:,j]),j] = np.argmax(t,axis=1)
                    emap[np.invert(continu_complete_case[:,j]),j] = m[j][-1]
                    
                    # On calcule e^j | reste
                    if (np.invert(continu_complete_case[:,j]).sum() == 0):
                        t = t*y_p
                    else:
                        t = t[:,0:(len(m[j])-1)]*y_p[continu_complete_case[:,j],0:(len(m[j])-1)]
                         
                    t = t/(t.sum(axis=1)[:,None])
                    
                    # On met à jour e^j
                    edisc[continu_complete_case[:,j],j] = vectorized(t, m[j])
                    edisc[np.invert(continu_complete_case[:,j]),j] = max(m[j])
                    
                # Variables qualitatives
                else:
                    # On fait le tableau de contingence e^j | x^j
                    link[j] = Counter([tuple(element) for element in np.column_stack((predictors_trans[train,j-d1],edisc[train,j]))])
            
                    y_p = np.zeros((n,len(m[j])))
                    
                    # On calcule y | e^{-j} , e^j
                    for k in range(len(m[j])):
                        modalites = np.zeros((n,len(m[j])))
                        modalites[:,k] = np.ones((n,))
            
                        y_p[:,k] = model_edisc.predict_proba(np.column_stack((base_disjonctive[:,0:(sum(list(map(len,m[0:j]))))],modalites,base_disjonctive[:,(sum(list(map(len,m[0:(j+1)])))):(sum(list(map(len,m))))])))[:,1]*(2*np.ravel(self.labels)-1)-np.ravel(self.labels)+1        
            
            
                    t = np.zeros((n,int(len(m[j]))))
                    
                    # On calcule e^j | x^j sur tout le monde
                    for l in range(n):
                        for k in range(int(len(m[j]))):
                            t[l,k] = link[j][(predictors_trans[l,j-d1],k)]/n
    
                    # On met à jour emap^j  
                    emap[:,j] = np.argmax(t,axis=1)
                    
                    # On calcule e^j | reste
                    t = t*y_p
                    t = t/(t.sum(axis=1)[:,None])
                    
                    edisc[:,j] = vectorized(t, m[j])
    
            # On regarde si des modalités sont présentes dans validation et pas dans train
            
    
            
        ################## Fin des itérations MCMC ##################
            
        # Meilleur(s) modèle(s) et équation de régression logistique
        
        
        
        
        # Evaluation de la performance
        
        
        
        
        
    
    def bestFormula(self):
        """Returns the best formula found by the MCMC."""
        emap_best = self.discretize(self.predictors_cont,self.predictors_qual)
        
        # Traitement des variables continues
        
        # Calculate shape of predictors (re-used multiple times)
        try:
            d1 = self.predictors_cont.shape[1]
        except AttributeError:
            d1 = 0
    
        try:
            d2 = self.predictors_qual.shape[1]
        except AttributeError:
            d2 = 0
        
        best_disc = []
        
        for j in range(d1):
            best_disc.append([])
            for k in np.unique(emap_best[:,j]):
                best_disc[j].append(np.nanmin(self.predictors_cont[emap_best[:,j]==k,j]))
            del best_disc[j][np.where(best_disc[j]==np.nanmin(best_disc[j]))[0][0]]
            print("Cut-points found for continuous variable",j+1,"are",best_disc[j])
                
        for j in range(d2):
            best_disc.append([])
            for k in np.unique(emap_best[:,j+d1]):
                best_disc[j+d1].append(np.unique(self.predictors_qual[emap_best[:,j+d1]==k,j]))
            print("Regroupments made for categorical variable",j+1,"are",best_disc[j+d1])

        return best_disc
    
    def performance(self):
        """Returns the best performance found by the MCMC."""

        return 0
    
    def discreteData(self):
        """Returns the best discrete data found by the MCMC."""
        return 0
    
    def contData(self):
        """Returns the continuous data provided to the MCMC as a single pandas dataframe."""
        return [self.predictors_cont, self.predictors_qual, self.labels]
    
    def discretize(self,predictors_cont,predictors_qual):
        """Discretizes new continuous and categorical features using a previously fitted glmdisc object.
        
        Keyword arguments:
        predictors_cont -- Continuous predictors to be discretized in a numpy "numeric" array. Can be provided either here or with the __init__ method.
        predictors_qual -- Categorical features which levels are to be merged (also in a numpy "string" array). Can be provided either here or with the __init__ method.
        """

        try:
            n = predictors_cont.shape[0]
        except AttributeError:
            n = predictors_qual.shape[0]
    
        try:
            d_1 = predictors_cont.shape[1]
        except AttributeError:
            d_1 = 0
    
        try:
            d_2 = predictors_qual.shape[1]
        except AttributeError:
            d_2 = 0

        d_1bis = [isinstance(x,sk.linear_model.logistic.LogisticRegression) for x in self.best_link]
        d_2bis = [isinstance(x,Counter) for x in self.best_link]
        
        if d_1!=sum(d_1bis): raise ValueError('Shape of predictors1 does not match provided link function')
        if d_2!=sum(d_2bis): raise ValueError('Shape of predictors2 does not match provided link function')
        
        emap = np.array([0]*n*(d_1+d_2)).reshape(n,d_1+d_2)

        for j in range(d_1+d_2):
            if d_1bis[j]:
                emap[np.invert(np.isnan(predictors_cont[:,j])),j] = np.argmax(self.best_link[j].predict_proba(predictors_cont[np.invert(np.isnan(predictors_cont[:,j])),j].reshape(-1,1)),axis=1)
                emap[np.isnan(predictors_cont[:,j]),j] = stats.describe(emap[:,j]).minmax[1] + 1
            elif d_2bis[j]:
                m = max(self.best_link[j].keys(),key=lambda key: key[1])[1]
                t = np.zeros((n,int(m)+1))
                
                for l in range(n):
                    for k in range(int(m)+1):
                        t[l,k] = self.best_link[j][(int((self.affectations[j].transform(np.ravel(predictors_qual[l,j-d_1])))),k)]/n
            
                emap[:,j] = np.argmax(t,axis=1)
                
            else: raise ValueError('Not quantitative nor qualitative?')
            
        return emap
    
    def discretizeDummy(self,predictors_cont,predictors_qual):
        """Discretizes new continuous and categorical features using a previously fitted glmdisc object as Dummy Variables usable with the best_reglog object.
        
        Keyword arguments:
        predictors_cont -- Continuous predictors to be discretized in a numpy "numeric" array. Can be provided either here or with the __init__ method.
        predictors_qual -- Categorical features which levels are to be merged (also in a numpy "string" array). Can be provided either here or with the __init__ method.
        """

        return self.best_encoder_emap.transform(self.discretize(predictors_cont,predictors_qual).astype(str))
    
    
    def predict(self,predictors_cont,predictors_qual):
        """Predicts the label values with new continuous and categorical features using a previously fitted glmdisc object.
        
        Keyword arguments:
        predictors_cont -- Continuous predictors to be discretized in a numpy "numeric" array. Can be provided either here or with the __init__ method.
        predictors_qual -- Categorical features which levels are to be merged (also in a numpy "string" array). Can be provided either here or with the __init__ method.
        """

        return self.best_reglog.predict_proba(self.discretizeDummy(predictors_cont,predictors_qual))


## Faire un try catch pour warm start ?

    


    def plot(self,predictors_cont_number="all",predictors_qual_number="all",plot_type="disc"):
        """Plots the stepwise function associating the continuous features to their discretization, the groupings made and the interactions.
        
        Keyword arguments:
        predictors_cont_number -- Which continuous variable(s) should be plotted (between 1 and the number of columns in predictors_cont).
        predictors_qual_number -- Which categorical variable(s) should be plotted (between 1 and the number of columns in predictors_qual).
        """
        emap = self.discretize(self.predictors_cont,self.predictors_qual).astype(str)
        try:
            d1 = self.predictors_cont.shape[1]
        except AttributeError:
            d1 = 0
    
        try:
            d2 = self.predictors_qual.shape[1]
        except AttributeError:
            d2 = 0


        if plot_type=="disc":
    
            if predictors_cont_number=="all":
                for j in range(d1):
                    plt.plot(self.predictors_cont[:,j].reshape(-1,1),emap.astype(str)[:,j],'ro')
                    plt.show()
    
            if predictors_qual_number=="all":
                for j in range(d2):
                    plt.plot(self.predictors_qual[:,j].reshape(-1,1),emap.astype(str)[:,j+d1],'ro')
                    plt.show()
    
            if not predictors_cont_number =="all":
                if (type(predictors_cont_number)==int and predictors_cont_number>0):
                    plt.plot(self.predictors_cont[:,predictors_cont_number-1].reshape(-1,1),emap.astype(str)[:,predictors_cont_number-1],'ro')
                    plt.show()
                else: warnings.warn('A single int (more than 0 and less than the number of columns in predictors_cont) must be provided for predictors_cont_number')
                
                
            if not predictors_qual_number =="all":
                if (type(predictors_qual_number)==int and predictors_qual_number>0):
                    plt.plot(self.predictors_qual[:,predictors_qual_number-1].reshape(-1,1),emap.astype(str)[:,predictors_qual_number-1+d1],'ro')
                    plt.show()
                else: warnings.warn('A single int (more than 0 and less than the number of columns in predictors_qual) must be provided for predictors_qual_number')

        elif plot_type=="logodd":
            
            # Gérer les manquants dans le GAM
            lignes_completes = np.invert(np.isnan(self.predictors_cont).sum(axis=1).astype(bool))

            # Fit du GAM sur tout le monde
            gam = LogisticGAM(dtype=['numerical' for j in range(d1)] + ['categorical' for d in range(d2)]).fit(pd.concat([pd.DataFrame(self.predictors_cont[lignes_completes,:]).apply(lambda x: x.astype('float')),pd.DataFrame(self.predictors_qual[lignes_completes,:]).apply(lambda x: x.astype('category'))],axis=1),self.labels[lignes_completes])
            
            # Quel que soit les valeurs de predictors_cont_number et predictors_qual_number, on plot tout pour l'instant
            XX = generate_X_grid(gam)
            plt.rcParams['figure.figsize'] = (28, 8)
            fig, axs = plt.subplots(1, d1+d2)
            for i, ax in enumerate(axs):
                pdep, confi = gam.partial_dependence(XX, feature=i+1, width=.95)
                ax.plot(XX[:, i], pdep)
                ax.plot(XX[:, i], confi[0][:, 0], c='grey', ls='--')
                ax.plot(XX[:, i], confi[0][:, 1], c='grey', ls='--')
                ax.plot(XX[:, i], )
            plt.show()
            
#            XX = generate_X_grid(gam)
#            plt.rcParams['figure.figsize'] = (28, 8)
#            fig, axs = plt.subplots(1, d1+d2)
#            for i, ax in enumerate(axs):
#                # Faire ici les graphiques du truc discret...
#                
#                ax.plot(XX[:, i], pdep)
#                ax.plot(XX[:, i], confi[0][:, 0], c='grey', ls='--')
#                ax.plot(XX[:, i], confi[0][:, 1], c='grey', ls='--')
#                ax.plot(XX[:, i], )
#            plt.show()
            
            
        
        

        
        
        
        
        
        