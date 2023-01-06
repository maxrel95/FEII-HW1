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

riskFreeRate = ( 1 / beta ) * np.exp(
    gamma * muC - 0.5 * gamma**2 * sigmaC** 2 ) - 1
f = A / ( 1 - A )
R = ( 1 + f ) / f

print( "RiskfreeRate", round(riskFreeRate*100, 2),"%\n",
       "P/E", round(f, 2), "\n",
       "Risky", round( (R-1)*100,2), "%" )

pd.DataFrame(
    [round(riskFreeRate*100, 2), round(f, 2), round( (R-1)*100,2)],
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
    R = (1 + f) / f
    return R - riskFreeRate

gamma = np.arange( 0, 250, 1 )
ERP = equityRiskPremium( gamma = gamma )

plt.figure()
plt.plot( gamma, ERP )
plt.title( "Equity Risk Premium as a function of risk aversion" )
plt.xlabel( "gamma" )
plt.ylabel( "E[R]-Rf" )
plt.savefig( "results/ERPGamma.png", dpi=1200 )
plt.show()

# Exo 3
df = pd.read_csv( "GMMData.csv", index_col=0, parse_dates=True )
data = pd.DataFrame()

data[ "realConsGrowth" ] = np.log(df[ 'Consumption' ] / df[ 'CPI' ]).diff().dropna()
data[ "RealSP500" ] = ((df[ "SP500" ] + df[ "Dividend" ])/df[ "CPI" ]).pct_change().dropna()
data[ 'realIr' ] = (1+df[ 'LTIR' ] / 100)**(1/12) / (1+df[ 'CPI' ].pct_change().dropna()) - 1

stat = pd.DataFrame(
    [ data.mean( 0 ), data.std( 0 ), data.skew( 0 ), data.kurt( 0 ) ],
    index=[ "Mean", "Standard Dev.", "Skewness", "Kurtosis" ]
)
stat.to_latex("results/ex3.tex")
corr = data.corr()
corr.to_latex("results/corr.tex")

plt.figure()
plt.title( "Correlation heatmap" )
sns.heatmap( corr, annot=True )
plt.savefig( "correlation.png", dpi=1200 )
plt.show()

plt.figure()
plt.plot( data )
plt.legend(data.columns.tolist())
plt.savefig("results/data.png")
plt.show()

plt.figure()
plt.plot( (1+data["RealSP500"]).cumprod())
plt.title("S&P500")
plt.savefig("results/sp500.png")
plt.show()

plt.plot( data["realConsGrowth"].cumsum())
plt.title("LogConsumption")
plt.savefig("results/cons.png")
plt.show()

plt.figure()
plt.plot((1+data["realIr"])**(12)-1)
plt.title("Real Interest Rate")
plt.savefig("results/reinte.png")
plt.show()

plt.figure()
plt.plot(df["LTIR"])
plt.title("Nominal Interest Rate")
plt.savefig("results/nomIR.png")
plt.show()

plt.figure()
plt.plot(df["CPI"].pct_change()*100)
plt.title("Inflation rate")
plt.savefig("results/inflation.png")
plt.show()
