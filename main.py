# Financial Econometrics II
# Homework I
# Author : Maxime Borel

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Exo 2
muC, muD = 1.89 / 100, 1.89 / 100
sigmaC, sigmaD = 1.5 / 100, 11.2 / 100
rho = 0.2
gamma = 2
beta = 0.98

A = beta * np.exp( muD - gamma * muC + 0.5 *
                   ( ( sigmaD * rho - sigmaC * gamma )**2 + sigmaD**2 * ( 1 - rho**2 ) ) )

riskFreeRate = ( 1 / beta ) * np.exp( gamma * muC - 0.5 * gamma**2 * sigmaC** 2 ) - 1
f = A / ( 1 - A )
R = ( ( 1 + f ) / f ) * np.exp( muD + 0.5*( sigmaD**2 ) )

print( "RiskfreeRate", round(riskFreeRate*100, 3),"%\n",
       "P/E", round(f, 3), "\n",
       "Risky", round( (R-1)*100,3), "%" )

pd.DataFrame(
    [round( riskFreeRate*100, 2 ), round( f, 2 ), round( ( R-1 )*100,2 ) ],
    index=["RiskfreeRate", "P/E", "ERP"]
).to_latex("results/ex1.tex")

def equityRiskPremium( gamma ):
    muC, muD = 1.89 / 100, 1.89 / 100
    sigmaC, sigmaD = 1.5 / 100, 11.2 / 100
    rho = 0.2
    beta = 0.98

    A = beta * np.exp( muD - gamma * muC + 0.5 *
                      ( ( sigmaD * rho - sigmaC * gamma ) ** 2 + sigmaD ** 2 * ( 1 - rho ** 2 ) ) )

    riskFreeRate = ( 1 / beta ) * np.exp( gamma * muC - 0.5 * gamma ** 2 * sigmaC ** 2 ) - 1
    f = A / (1 - A)
    R = ( ( 1 + f ) / f ) * np.exp( muD + 0.5*( sigmaD**2 ) ) - 1
    return R - riskFreeRate

def equityRateReturn( gamma ):
    muC, muD = 1.89 / 100, 1.89 / 100
    sigmaC, sigmaD = 1.5 / 100, 11.2 / 100
    rho = 0.2
    beta = 0.98

    A = beta * np.exp( muD - gamma * muC + 0.5 *
                      ( ( sigmaD * rho - sigmaC * gamma ) ** 2 + sigmaD ** 2 * ( 1 - rho ** 2 ) ) )

    f = A / (1 - A)
    R = ( ( 1 + f ) / f ) * np.exp( muD + 0.5*( sigmaD**2 ) ) - 1
    return R 

gamma = np.arange( 0, 50, 1 )
ERP = equityRiskPremium( gamma = gamma )
Risky = equityRateReturn( gamma=gamma )

plt.figure(1)
plt.plot( gamma, ERP*100 )
plt.title( "Equity Risk Premium as a function of risk aversion" )
plt.xlabel( "gamma" )
plt.ylabel( "E[R]-Rf in (%)" )
plt.savefig( "results/ERPGamma.png", dpi=1200 )
plt.show()

plt.figure(2)
plt.plot( gamma, ERP*100 )
plt.plot( gamma, Risky*100 )
plt.title( "Equity Risk Premium as a function of risk aversion" )
plt.legend(["ERP", "Risky"])
plt.xlabel( "gamma" )
plt.ylabel( "Rate of Return in (%)" )
plt.savefig( "results/ERPRiskyGamma.png", dpi=1200 )
plt.show()

# Exo 3
df = pd.read_csv( "GMMData.csv", index_col=0, parse_dates=True )
data = pd.DataFrame()

data[ "realConsGrowth" ] = np.log(df[ 'Consumption' ] / df[ 'CPI' ]).diff().dropna()
data[ "RealSP500" ] = ((df[ "SP500" ] + df[ "Dividend" ])/df[ "CPI" ]).pct_change().dropna()
data[ 'realIr' ] = (1+df[ 'LTIR' ] / 100)**(1/12) / (1+df[ 'CPI' ].pct_change().dropna()) - 1

stat = pd.DataFrame(
    [ data.mean( 0 )*1200, data.std( 0 )*np.sqrt(12)*100, data.skew( 0 ), data.kurt( 0 ) ],
    index=[ "Mean", "Standard Dev.", "Skewness", "Kurtosis" ]
).round( 3 )
stat.to_latex("results/ex3.tex")

corr = data.corr()
corr.to_latex("results/corr.tex")

plt.figure(3)
plt.title( "Correlation heatmap" )
sns.heatmap( corr, annot=True, cmap="ocean" )
plt.savefig( "results/correlation.png", dpi=1200 )
plt.show()

plt.figure(4)
plt.plot( (1+data["RealSP500"]).cumprod()*100)
plt.title("Real S&P500")
plt.ylabel("$")
plt.savefig("results/sp500.png", dpi=1200)
plt.show()

plt.plot( data["realConsGrowth"].cumsum())
plt.title("Cumulative Real Log Consumption")
plt.savefig("results/cons.png", dpi=1200)
plt.show()

plt.figure(5)
plt.plot((1+data["realIr"])**(12)-1)
plt.title("Real Interest Rate")
plt.ylabel("R in (%)")
plt.savefig("results/reinte.png", dpi=1200)
plt.show()

plt.figure(5)
plt.plot((1+data["realIr"]).cumprod())
plt.title("Cumulative investment in Real Interest Rate")
plt.ylabel("$")
plt.savefig("results/cumprodREINTE.png", dpi=1200)
plt.show()

plt.figure(6)
plt.plot(df["CPI"].pct_change()*100)
plt.title("Inflation rate")
plt.ylabel("Inflation rate in %")
plt.savefig("results/inflation.png", dpi=1200)
plt.show()

plt.figure(7)
(df["Dividend"]+df["SP500"]).plot()
plt.title("Nominal S&P 500")
plt.savefig("results/sp500nom.png", dpi=1200)
plt.ylabel("Price")
plt.show()

plt.figure(8)
df["LTIR"].plot()
plt.ylabel("R in (%)")
plt.title("Nominal Interest Rate")
plt.savefig("results/nomIR.png", dpi=1200)
plt.show()

plt.figure(9)
df["CPI"].plot()
plt.ylabel("Price")
plt.title("consumer price index")
plt.savefig("results/CPI.png", dpi=1200)
plt.show()

plt.figure(10)
df["Consumption"].plot()
plt.ylabel("Price")
plt.title("Nominal consumption")
plt.savefig("results/nomConsumption.png", dpi=1200)
plt.show()
