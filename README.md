# Cryptocurrency Momentum and Pairs Trading with Sentiment Analysis

## Introduction

Cryptocurrency markets are extremely **volatile** and **sentiment-driven**. This allows for a wide range of quantative strategies. In this project I go through three approaches.

1. **Cross-Sectional Momentum**: Ranking winners and losers among the top 5 cryptocurrencies (BTC, ETH, ADA, BNB, XRP) based on recent performance.
2. **Statistical Arbitrage (Pairs Trading)**: A market-neutral strategy using the correlation of Ethereum (ETH) and Bitcoin (BTC)
3. **Sentiment-Enhanced Signals**: Using Twitter and Google trends to refine search trends

We have multiple years of daily data (2020–2024) and my work shows that you can combine both traditional and modern approaches such as cross sectional momentum and sentiment.

## Data and Methodology

### Data Sources and Processing

We obtained daily price data (open/close prices) for five major cryptocurrencies (Note: I should probably expand this to avoid any overfitting in rank):
- **Bitcoin (BTC)**
- **Ethereum (ETH)**
- **Cardano (ADA)**
- **Binance Coin (BNB)**
- **Ripple (XRP)**

The data through 2020-2024 has been both bullish and bearish so these coins on this date range is very acceptable. To see how I loaded the data feel free to look in the notebooks.

For sentiment analysis, I gathered two external datasets:

1. **Google Trends** search interest for cryptocurrency-related keywords (ex. "Bitcoin"). The Google Trends index (0–100) is a sentiment score
2. **Twitter Sentiment** data via the VADER sentiment analysis tool. I used a kaggle dataset of a bunch of tweets  mentioning major crypto terms and computed a daily average sentiment score (ranging from -1 to +1) to guage the public mood

We combined these into a composite sentiment index by standardizing each metric and averaging. This composite index identifies high sentiment regimes (when optimism and attention are high) versus low sentiment regimes.

### Cross-Sectional Momentum Strategy

**Momentum hypothesis**: Assets that have outperformed their peers recently may continue to outperform, and the vice versa applies

**Signal construction**: We define each coin's momentum as its trailing X-day return (ex. past 30-day percentage price change). Each day, all five coins are ranked by this momentum value from highest to lowest.

**Portfolio formation**: At each rebalancing interval (daily in our analysis), we go long the top-ranked momentum coin(s) and short the bottom-ranked coin(s).

```python
# Pseudocode for daily cross-sectional momentum strategy
lookback_days = 30  

momentum_df = pd.DataFrame(index=all_coins['BTC'].index)

for coin, df in all_coins.items():
    coin_symbol = coin.replace('USDT','')
    momentum_df[coin_symbol] = df['close'].pct_change(lookback_days)

ranks = momentum_df.rank(axis=1, ascending=False, method='min')

num_coins = len(momentum_df.columns)
top_pct = 0.4  

top_cutoff = int(num_coins * top_pct)
if top_cutoff == 0:
    top_cutoff = 1  # Always at least one coin

long_positions = ranks <= top_cutoff

```

Figure: Cross-Sectional Momentum Strategy vs. Bitcoin Buy-and-Hold (2020–2024)

This chart compares the cumulative returns of our cross-sectional momentum strategy (blue line) against a simple Bitcoin buy-and-hold benchmark (orange line). Initially, both strategies closely track each other, but during significant bull market phases like late 2024, Bitcoin’s buy-and-hold outperformed temporarily, benefiting from large upward price movements. However, during volatile or bear-market periods (such as throughout 2022–2023), the momentum strategy performed very well, with less volatility and less drawdown. The momentum strategy is mildly effective with some lag.

![Screenshot 2025-07-05 at 10 12 58 PM](https://github.com/user-attachments/assets/dbea774b-4d54-412b-9f80-99feb174b8bb)


### ETH–BTC Pairs Trading Strategy

**Strategy concept**: Pairs trading is a much more stable, market-neutral approach by focusing on relationships instaed of actual price

**Correlation test**: We first test whether ETH and BTC prices are cointegrated using an Augmented Dickey-Fuller (ADF) test on the log-price spread:

```
Spread(t) = pW(ETH(t)) - β · pL(Price_BTC(t))
```

where β is a hedge ratio estimated that was found via linear regression of ETH on BTC. A non-moving spread means any divergence between ETH and BTC eventually mean-reverts (this is really important).

**Trading rules**: We convert the spread into a Z-score (number of standard deviations away from its historical mean). When the spread's Z-score exceeds a threshold (ex. +2σ), it indicates ETH is relatively overpriced vs BTC. With this the strategy then shorts ETH and buys BTC expecting mean reversion

```python
# Linear Regression for β 
X = btc_log.values.reshape(-1, 1)
X = sm.add_constant(X)  # adds intercept
model = sm.OLS(eth_log, X).fit()

hedge_ratio = model.params[1]
print("Hedge ratio ETH/BTC:", hedge_ratio)

lookback = 60  # days for rolling stats

spread_mean = spread.rolling(lookback).mean()
spread_std = spread.rolling(lookback).std()

z_score = (spread - spread_mean) / spread_std

long_spread = z_score < -1
short_spread = z_score > 1

# Signal: +1 for long, -1 for short, 0 otherwise
pairs_signal = np.where(long_spread, 1, np.where(short_spread, -1, 0))
signals_df = pd.Series(pairs_signal, index=z_score.index)
spread_ret = eth_log.diff() - hedge_ratio * btc_log.diff()
pairs_pnl = signals_df.shift(1) * spread_ret
pairs_cum = (1 + pairs_pnl.fillna(0)).cumprod()


```
Figure: ETH/BTC Pairs Trading Backtest (2020–2024)

This chart shows the cumulative performance of the ETH/BTC pairs trading strategy over the testing period. Initially, the strategy looks really promising. However, performance a LOT from 2021 onwards, stabilizing  after mid-2021 and continuing to drift downwards through 2024.

This shows a key insight in the crupto coins I've missed: despite ETH and BTC being statistically cointegrated and theoretically suitable for pairs trading, a lot of movements in the crypto industry (market regime shifts, fundamental changes, or prolonged trends) significantly weakened the effectiveness of a straightforward mean-reversion strategy. For practical trading purposes, more adjustments need to be made. Which is why I moved on to sentiment. To see if that could look and see this market movement. 

![Screenshot 2025-07-05 at 10 21 29 PM](https://github.com/user-attachments/assets/708cfec4-c2f7-42eb-80f5-a0b8ba12e9b3)

Overall though, if you look in my notebooks, when I combined both pairs trading and momentum signals with certain weighting I got a sharpe of **1.1275915754553112**, which is fairly solid. I do think just overall momentum had a higher sharpe due to the crazy patterns in 2021 and 2024. Overall, a very solid strategy.

### Sentiment-Enhanced Momentum Signals

**Rationale**: Crypto markets are highly sentiment-driven. For example, my two crypto purchases have been when I heard something from my friends about it. (not wall street bets at least lol)

**Composite sentiment index**: Using the Google Trends and Twitter data described earlier, I created a daily composite sentiment index. I then defined:

- **High Sentiment Regime**: Days when the sentiment index is in the top quartile of its historical range
- **Low Sentiment Regime**: Days in the bottom quartile of sentiment

**Conditional strategy**: We adjust the momentum strategy based on sentiment:

```python
# This is just psuedocode as my actual code is way too long to be shown here. Please check crypto-twitter for how I incorporated this
for day in trading_days:
    if sentiment_index[day] > high_threshold:
        weights = generate_momentum_signal(day)  # long top coin, short bottom coin
    elif sentiment_index[day] < low_threshold:
        weights = {}  # no position (risk-off during low sentiment)
    else:
        weights = generate_momentum_signal(day) 
    
    execute_trade(weights)
```

Google Sentiment Patterns on Momentum

![Screenshot 2025-07-05 at 10 41 17 PM](https://github.com/user-attachments/assets/cca1ef5b-658f-49c0-b9d3-ce045612c552)

You can see the clear correlation from this graph. Low correlation leads to less returns. Now let's interpret all of our results and backtest this


## Results and Analysis


### Cross-Sectional Momentum Signal Analysis



![Screenshot 2025-07-05 at 10 37 49 PM](https://github.com/user-attachments/assets/650d41fd-5b70-4acc-875d-da1d09298302)

**Figure**: Average Next-Day Return by Momentum Rank (Top 5 Cryptocurrencies)

This analysis confirms a classic momentum effect in the cross-section of cryptocurrencies. We sorted the five assets by their daily momentum rank and computed the average of their next-day returns:

- **Rank 1** (highest momentum): Largest mean next-day return
- **Rank 5** (lowest momentum): Near-zero or minimal subsequent returns
The nearly linear drop from Rank 1 to Rank 5 implies the momentum signal contains genuine predictive information rather than random noise.

![Screenshot 2025-07-05 at 10 38 19 PM](https://github.com/user-attachments/assets/6c0df06f-95ef-44ca-b72f-a7dbe4cc046d)

**Figure**: Trailing Return Spread (Top 1 vs Bottom 1 Momentum Asset)

This time-series plot tracks the difference in recent performance between the best and worst performing coin at each moment. Key observations:

- Higher spreads indicate fertile situations for momentum strategies
- In early 2021, the spread spiked dramatically (5-6x difference in 30-day returns)
- During late 2022, the spread briefly turned negative, hurting momentum strategies

  
![Screenshot 2025-07-05 at 10 38 32 PM](https://github.com/user-attachments/assets/0a82c44a-fc66-4868-a28c-88e9ce8e10be)

**Figure**: 30-Day Rolling Sharpe Ratio – Momentum Factor Portfolio

The momentum strategy's Sharpe ratio has been highly variable:

- **Peak periods**: Extraordinarily high Sharpe values (above 5-10) during strong trending periods
- **Challenging periods**: Negative Sharpe ratios when trends reversed or all coins fell in tandem
- What I get from this: Momentum is not consistently reliable clearly

### ETH–BTC Pairs Trading Results

![Screenshot 2025-07-05 at 10 39 04 PM](https://github.com/user-attachments/assets/a7d716a8-aa97-4bc9-a394-24c382cc793a)

**Figure**: ETH/BTC Price Spread – Z-Score of Cointegrated Relationship

The standardized spread between Ethereum and Bitcoin prices confirms a stationary, mean-reverting relationship:

- Z-score of 0 means ETH/BTC is at historical equilibrium
- Positive values mean ETH is expensive vs. BTC
- Negative values mean ETH is cheap vs. BTC

The spread typically stays within ±1 standard deviations but has some notable deviations

![Screenshot 2025-07-05 at 10 39 28 PM](https://github.com/user-attachments/assets/2cb1e565-7ea3-4ad6-a50b-c5bac8538b5b)

**Figure**: ETH/BTC Pairs Trading Strategy – Cumulative P&L

The pairs trading strategy showed mixed results:

- **Initial success**: Profitable in late 2020 through early 2021
- **Significant decline**: 40% loss by mid-2021 when trends overpowered mean reversion
- **Overall outcome**: Mid haha


### Sentiment Analysis & Momentum Integration

![Screenshot 2025-07-05 at 10 40 10 PM](https://github.com/user-attachments/assets/6517d4f6-1a76-4c78-9464-422d2efc43ce)

**Figure**: Daily Composite Sentiment vs. Next-Day Momentum Return (Scatter Plot)

The scatter plot reveals a fairly diffuse cloud with no strong linear pattern, indicating that daily sentiment alone doesn't reliably predict next-day momentum returns. While there was a lot of correlation in twitter data and google trend data to the same day. It tended to revert/change a lot the next day which skewed a lot of the results. 


# Please look in my notebook for more graphs, code, explanations. I tried my best to explain the gist of it here.
