import yfinance as yf
import pandas as pd
import argparse

def stock_data(
    ticker:str,
    start:str,
    end:str,
    period:str
    ):
    return yf.download(ticker, start, end, period)

if __name__=="__main__":

    parser = argparse.parArgumentParser()

    parser.add_argument("--ticker", help="Name of the ticker", type=str)
    parser.add_argument("--start", help="Start date in format 'YYYY-MM-DD'", type=str)
    parser.add_argument("--end", help="End date in format 'YYYY-MM-DD'", type=str)
    parser.add_argument("--freq", help="Frequency of data reqd., '1d','5d','1mo','3mo' etc.", type=str)

    args = parser.parse_args()

    return stock_data(ticker=args.ticker, start=args.start, end=args.end, period=args.freq)
