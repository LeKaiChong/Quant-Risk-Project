import yfinance as yf
from datetime import date
import matplotlib.pyplot as plt
import numpy as np 
import yaml
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
from sklearn.decomposition import PCA

class ConfigLoader():
    @staticmethod
    def configload(filepath="config.yaml"):
        with open(filepath, "r") as f:
            config = yaml.safe_load(f)
        return config

class data_extraction():
    @staticmethod
    def get_data(stocklist:list, startdate: date, enddate: date):
        stockData = yf.download(stocklist, start=startdate, end=enddate, auto_adjust=True)
        stockData = stockData['Close']
        returns = stockData.pct_change()
        meanReturns = returns.mean()
        covMatrix = returns.cov()
        return returns,meanReturns, covMatrix
    
class montecarlosims():
    
    def monte_carlo(timeframe:int,covmatrix:np.ndarray,number_of_sims:int,meanreturn:object,weightofport:np.ndarray,initialportfoliovalue:int ):
        portfolios_sims = np.full(shape=(timeframe, number_of_sims), fill_value=0.0)
        meanreturn_matrix= np.full(shape=(timeframe, len(weightofport)), fill_value=meanreturn)
        meanreturn_matrix=meanreturn_matrix.T
        for m in range(number_of_sims):
        # MC Loops
            Z = np.random.normal(size=(timeframe, len(weightofport)))
            L = np.linalg.cholesky(covmatrix)# This translates independant shocks into correlated shocks
            dailyreturns = meanreturn_matrix + np.inner(L,Z) # compute the inner product of random shock L and correlated shock Z
            portfolios_sims[:,m] = np.cumprod(np.inner(weightofport, dailyreturns.T)+1)*initialportfoliovalue # cumulative increase in return -> convert to portfolio value 

        #Graph sims
        plt.plot(portfolios_sims)
        plt.ylabel('Portfolio value ($)')
        plt.xlabel('Date')
        plt.title(' MC simulation of a stock portfolio')
        plt.show()

        #mean return
        paths = portfolios_sims              # (T, mc_sims)
        V0 = float(initialportfoliovalue)
        terminal = paths[-1, :]             # (mc_sims,)
        exp_terminal = terminal.mean()
        print("Expected return:", exp_terminal)

        #medium return
        median_terminal = np.median(terminal)
        print("Median return:", median_terminal)

        # prob that VT < V0
        loss_prob = np.mean(terminal < V0)
        print("Loss probability:", loss_prob)
        V0 = float(initialportfoliovalue)
        terminal = paths[-1, :]
        losses = V0 - terminal
        conf = 0.95  # 95% VaR
        var = np.quantile(losses, conf)
        cvar = losses[losses >= var].mean()

        print(f"VaR ({int(conf*100)}%): {var:.2f}")
        print(f"CVaR ({int(conf*100)}%): {cvar:.2f}")


class factor_models():

    @ staticmethod

    def preliminaryfactor_modelling(stockdata: pd.DataFrame, factor_Data: pd.DataFrame):
        results = {}
        residuals = pd.DataFrame(index=stockdata.index)
        # Ensure we have a constant in X (alpha)
        # We'll build X once per ticker after dropping RF
        for ticker in stockdata.columns:
            # y = excess return
            y = (stockdata[ticker] - factor_Data["RF"]).rename("y")

            # Merge and drop missing rows
            df = pd.concat([y, factor_Data], axis=1).dropna()

            y_clean = df["y"]

            # X = factors (exclude RF), add constant
            X_clean = df.drop(columns=["RF", "y"])
            X_clean = sm.add_constant(X_clean)

            model = sm.OLS(y_clean, X_clean).fit()
            results[ticker] = model
            residuals[ticker] = model.resid.reindex(stockdata.index)
            print(f"\n===== {ticker} =====")
            print(model.summary())

        # Compile beta table
        rows = []
        for ticker, model in results.items():
            rows.append({
                "Ticker": ticker,
                "Alpha": model.params.get("const"),
                "Beta_Mkt": model.params.get("SPY-RF"),
                "Beta_Tech": model.params.get("Tech"),
                "Beta_HML": model.params.get("HML"),
                "Beta_SMB": model.params.get("SMB"),
                "R2": model.rsquared,
                "N": int(model.nobs)
            })

        betas_df = pd.DataFrame(rows).set_index("Ticker")
        return betas_df,results,residuals
    

    @ staticmethod

    def refinefactor_modelling(stockdata: pd.DataFrame, factor_Data: pd.DataFrame):
        results = {}
        residuals = pd.DataFrame(index=stockdata.index)
        # Ensure we have a constant in X (alpha)
        # We'll build X once per ticker after dropping RF
        for ticker in stockdata.columns:
            # y = excess return
            y = (stockdata[ticker] - factor_Data["RF"]).rename("y")

            # Merge and drop missing rows
            df = pd.concat([y, factor_Data], axis=1).dropna()

            y_clean = df["y"]

            # X = factors (exclude RF), add constant
            X_clean = df.drop(columns=["RF", "y"])
            X_clean = sm.add_constant(X_clean)

            model = sm.OLS(y_clean, X_clean).fit()
            results[ticker] = model
            residuals[ticker] = model.resid.reindex(stockdata.index)
            print(f"\n===== {ticker} =====")
            print(model.summary())

        # Compile beta table
        rows = []
        for ticker, model in results.items():
            rows.append({
                "Ticker": ticker,
                "Alpha": model.params.get("const"),
                "Beta_Mkt": model.params.get("SPY-RF"),
                "Beta_Tech": model.params.get("Tech"),
                "Beta_HML": model.params.get("HML"),
                "Beta_SMB": model.params.get("SMB"),
                "Beta_defence":model.params.get("Defence_impact"),
                "Beta_speculation": model.params.get('Speculation_impact'),
                "Beta_Momentum": model.params.get("Momentum_impact"),
                "R2": model.rsquared,
                "N": int(model.nobs)
            })

        betas_df = pd.DataFrame(rows).set_index("Ticker")
        return betas_df,results,residuals
    
    def pca_residuals(residual_matrix):
        resid_clean = residual_matrix.dropna()
        pca = PCA()
        pca.fit(resid_clean)

        explained = pca.explained_variance_ratio_

        print("Explained variance ratios:")
        print(explained[:5])
        
        print('printing the components of pc1')
        print(pd.Series(pca.components_[0], index=resid_clean.columns).sort_values(ascending=False))


        print('printing the components of pc2')
        print(pd.Series(pca.components_[1], index=resid_clean.columns).sort_values(ascending=False))
        
    def visualise_residuals(residual_matrix:np.ndarray):
        resid_corr = residual_matrix.corr()
        
        plt.figure(figsize=(12,6))
        sns.heatmap(resid_corr, annot=True, cmap="coolwarm", center=0,)
        plt.title("Residual Correlation Matrix")
        plt.show()

    @staticmethod
    def factor_model_monte_carlo(betas_df,
        factor_df,
        residual_df,
        weights,
        initial_value=10000,
        timeframe=100,
        n_sims=5000,
        rf_daily=0.0,
        seed=42):
        rng = np.random.default_rng(seed)

        # alpha vector
        alpha = betas_df["Alpha"].copy()

        # beta matrix
        B = betas_df[
            [
                "Beta_Mkt",
                "Beta_Tech",
                "Beta_HML",
                "Beta_SMB",
                "Beta_defence",
                "Beta_speculation",
                "Beta_Momentum",
            ]
        ].copy()

        B.columns = [
            "SPY-RF",
            "Tech",
            "HML",
            "SMB",
            "Defence_impact",
            "Speculation_impact",
            "Momentum_impact",
        ]

        # align order
        tickers = list(B.index)
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()

        # factor moments
        factor_df = factor_df[B.columns].copy()
        mu_f = factor_df.mean().values # compute the average return of each factor
        sigma_f = factor_df.cov().values # get the covariance matrix of the factors
        L_f = np.linalg.cholesky(sigma_f + 1e-12 * np.eye(len(mu_f))) # adds a tiny number in the digaonal

        # residual covariance
        residual_df = residual_df[tickers].copy()
        sigma_eps = residual_df.cov().values
        L_eps = np.linalg.cholesky(sigma_eps + 1e-12 * np.eye(len(tickers)))

        paths = np.zeros((timeframe, n_sims))
        terminal = np.zeros(n_sims)

        for m in range(n_sims):
            # simulate factor returns
            Z_f = rng.standard_normal(size=(timeframe, len(mu_f)))
            F_sim = Z_f @ L_f.T + mu_f

            # simulate residual shocks
            Z_eps = rng.standard_normal(size=(timeframe, len(tickers)))
            eps_sim = Z_eps @ L_eps.T # adds the factor covariance effect into the random simulation

            # rebuild stock excess returns
            stock_excess = F_sim @ B.values.T + alpha.values + eps_sim

            # add risk-free rate
            stock_total = stock_excess + rf_daily

            # portfolio returns
            port_ret = stock_total @ weights

            # portfolio path
            V = initial_value * np.cumprod(1 + port_ret)
            paths[:, m] = V
            terminal[m] = V[-1]

        losses = initial_value - terminal
        var95 = np.quantile(losses, 0.95)
        cvar95 = losses[losses >= var95].mean()
        loss_prob = np.mean(terminal < initial_value)

        return {
            "paths": paths,
            "terminal": terminal,
            "expected_terminal": terminal.mean(),
            "median_terminal": np.median(terminal),
            "loss_probability": loss_prob,
            "VaR_95": var95,
            "CVaR_95": cvar95,
        }
    @staticmethod
    def stress_testing(summary_df: pd.DataFrame, shock):

        # Extract alpha
        alpha = summary_df.loc["Portfolio", "Alpha"]

        # Extract beta estimates
        betas = summary_df.loc["Portfolio", [
            "Beta_Mkt",
            "Beta_Tech",
            "Beta_HML",
            "Beta_SMB",
            "Beta_defence",
            "Beta_speculation",
            "Beta_Momentum"
        ]].copy()

        # Rename to factor names
        betas.index = [
            "SPY-RF",
            "Tech",
            "HML",
            "SMB",
            "Defence_impact",
            "Speculation_impact",
            "Momentum_impact"
        ]

        shock = pd.Series(shock, dtype=float)

        common = betas.index.intersection(shock.index)
        contribution = betas.loc[common] * shock.loc[common]

        details = pd.DataFrame({
            "beta": betas.loc[common],
            "shock": shock.loc[common],
            "contribution": contribution
        })

        portfolio_return = alpha + contribution.sum()
        return portfolio_return, details.sort_values("contribution")