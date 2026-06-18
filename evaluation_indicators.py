# This file is to calculate selected model formance indicators
# Copyright (C) 2019  Haiyan Wei

import pandas as pd
import numpy as np


def get_indicators(sim, obs):

    """ call functsions to get all indicator values """

    PBIAS, RSR, R2, NS, RMSE, KGE, OF, OF_W = [np.float64('NaN')] * 8

    if not len(sim) == len(obs):
        msg = 'Error: simulated values and observed values do not have the same length.'

    else:
        df = pd.DataFrame([]).assign(sim=sim, obs=obs)
        df = df.dropna(how='any', axis=0)
        sim = df.sim.values
        obs = df.obs.values
        N = len(obs)
        if len(obs) <= 2:
            msg = 'Error: no enough points for model performance evaluation.'

        else:
            PBIAS = pbias(sim, obs)
            RSR = rsr(sim, obs)
            R2 = r2(sim, obs)
            NS = nse(sim, obs)
            RMSE = rmse(sim, obs)
            KGE = kge(sim, obs)
            OF = (1 - NS) + (1 - R2) + np.abs(PBIAS) / 100. + RMSE + RSR
            OF_W = (1 - NS) * 0.25 + (1 - R2)*0.5/3 + \
                np.abs(PBIAS) / 100.*0.25 + RMSE*0.5/3 + RSR*0.5/3
            msg = f'N={N}, R2={R2:.2f}, RMSE={RMSE:.2f}, NSE={NS:.2f}, RSR={RSR:.2f}, pbias={PBIAS:.2f}, KGE={KGE:.2f}, OF={OF:.2f}, OF_W={OF_W:.2f}'

    print(msg)
    return [msg, N, PBIAS, RSR, R2, NS, RMSE, KGE, OF, OF_W]



def kge(sim, obs):
    """ calculate KGE """
    r = np.corrcoef(sim, obs)[0, 1]
    gamma = (np.std(sim) / np.mean(sim)) / (np.std(obs) / np.mean(obs)) # 2012
    # gamma = (np.std(sim)) / (np.std(obs))  # 2009
    beta = np.mean(sim) / np.mean(obs)

    return 1 - np.sqrt((r - 1) ** 2 + (gamma - 1) ** 2 + (beta - 1) ** 2)


def rmse(sim, obs):
    """ calculate RMSE: Root Mean Square Error """
    return np.sqrt(np.mean((sim - obs)**2))


def nse(sim, obs):
    """calculate Nash-Sutcliffe Efficiency"""
    return 1 - sum((sim - obs)**2) / sum((obs - np.mean(obs))**2)


def r2(sim, obs):
    """ calculate r2 """
    return (np.corrcoef(sim, obs)[0, 1]) ** 2


def pbias(sim, obs):
    """ calculate PBIAS """
    return sum(obs-sim) / sum(obs) * 100.0


def rsr(sim, obs):
    """ calculate RSR """
    sse = sum((obs-sim) ** 2)
    rmse = np.sqrt(sse / len(obs))
    return rmse / np.std(obs)


# sim = [86.143,73.481,68.348,57.022,53.387,52.335,51.520,51.329,46.229,44.371]
# obs = [56.008,46.672,21.358,64.470,31.963,38.269,39.201,29.374,41.910,44.216]
# print(kge(sim, obs))
