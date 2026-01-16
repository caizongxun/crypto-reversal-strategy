#!/usr/bin/env python3

"""
Practical Reversal Detector - Based on Real Trading Setup

Method:
1. 在每根K棒執行交易（開多或開空）
2. 止損 = 1 ATR
3. 獲利目標 = 1.5 ATR (1:1.5風險報酬比)
4. 檢查是否獲利
5. 記錄獲利的K棒特徵
6. 找出獲利K棒的共同特徵 = 反轉點特徵
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
import warnings
warnings.filterwarnings('ignore')


class PracticalReversalTester:
    """Test trades on each candle using practical entry/exit rules."""
    
    def __init__(self, df: pd.DataFrame, atr_period: int = 14):
        """
        Initialize practical reversal tester.
        
        Args:
            df: DataFrame with OHLCV data
            atr_period: ATR計算週期
        """
        self.df = df.copy()
        self.atr_period = atr_period
        self.atr = self._calculate_atr(self.df, atr_period)
        self.df['ATR'] = self.atr
        
        # 交易結果
        self.trades = []  # 每筆交易的記錄
        self.winning_trades = []  # 獲利的交易
        self.losing_trades = []  # 虧損的交易
    
    @staticmethod
    def _calculate_atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
        """計算ATR。"""
        tr1 = df['High'] - df['Low']
        tr2 = abs(df['High'] - df['Close'].shift())
        tr3 = abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().fillna(0)
        return atr.values
    
    def backtest_long_from_candle(self, entry_idx: int, 
                                   lookforward_bars: int = 50) -> Dict:
        """
        在指定K棒開多單。
        
        Args:
            entry_idx: 入場K棒索引
            lookforward_bars: 往前看幾根K棒來計算結果
        
        Returns:
            交易結果字典
        """
        if entry_idx >= len(self.df) - lookforward_bars:
            return None
        
        entry_candle = self.df.iloc[entry_idx]
        entry_price = entry_candle['Close']
        atr_value = self.atr[entry_idx]
        
        if atr_value == 0:
            return None
        
        # 計算止損和獲利目標
        stop_loss = entry_price - atr_value  # 止損放1個ATR
        take_profit = entry_price + (atr_value * 1.5)  # 獲利目標 1.5 ATR (1:1.5)
        
        # 檢查接下來的K棒是否觸發止損或獲利
        future_highs = self.df.iloc[entry_idx+1:entry_idx+lookforward_bars+1]['High'].values
        future_lows = self.df.iloc[entry_idx+1:entry_idx+lookforward_bars+1]['Low'].values
        
        # 先觸發哪個？
        hit_tp_first = False
        hit_sl_first = False
        tp_bar = None
        sl_bar = None
        
        for i, (h, l) in enumerate(zip(future_highs, future_lows)):
            if l <= stop_loss:
                hit_sl_first = True
                sl_bar = entry_idx + 1 + i
                break
            if h >= take_profit:
                hit_tp_first = True
                tp_bar = entry_idx + 1 + i
                break
        
        # 計算結果
        if hit_tp_first:
            result = 'win'
            exit_price = take_profit
            profit = take_profit - entry_price
            profit_pct = (profit / entry_price) * 100
            exit_bar = tp_bar
        elif hit_sl_first:
            result = 'loss'
            exit_price = stop_loss
            profit = stop_loss - entry_price
            profit_pct = (profit / entry_price) * 100
            exit_bar = sl_bar
        else:
            result = 'no_exit'
            exit_price = self.df.iloc[entry_idx + lookforward_bars]['Close']
            profit = exit_price - entry_price
            profit_pct = (profit / entry_price) * 100
            exit_bar = entry_idx + lookforward_bars
        
        trade_info = {
            'entry_idx': entry_idx,
            'entry_price': entry_price,
            'entry_candle': entry_candle.to_dict(),
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'atr_value': atr_value,
            'direction': 'long',
            'result': result,
            'exit_price': exit_price,
            'exit_bar': exit_bar,
            'profit': profit,
            'profit_pct': profit_pct,
            'bars_held': (exit_bar - entry_idx) if exit_bar else lookforward_bars
        }
        
        return trade_info
    
    def backtest_short_from_candle(self, entry_idx: int,
                                    lookforward_bars: int = 50) -> Dict:
        """
        在指定K棒開空單。
        
        Args:
            entry_idx: 入場K棒索引
            lookforward_bars: 往前看幾根K棒來計算結果
        
        Returns:
            交易結果字典
        """
        if entry_idx >= len(self.df) - lookforward_bars:
            return None
        
        entry_candle = self.df.iloc[entry_idx]
        entry_price = entry_candle['Close']
        atr_value = self.atr[entry_idx]
        
        if atr_value == 0:
            return None
        
        # 計算止損和獲利目標
        stop_loss = entry_price + atr_value  # 止損放1個ATR
        take_profit = entry_price - (atr_value * 1.5)  # 獲利目標 1.5 ATR
        
        # 檢查接下來的K棒是否觸發止損或獲利
        future_highs = self.df.iloc[entry_idx+1:entry_idx+lookforward_bars+1]['High'].values
        future_lows = self.df.iloc[entry_idx+1:entry_idx+lookforward_bars+1]['Low'].values
        
        # 先觸發哪個？
        hit_tp_first = False
        hit_sl_first = False
        tp_bar = None
        sl_bar = None
        
        for i, (h, l) in enumerate(zip(future_highs, future_lows)):
            if h >= stop_loss:
                hit_sl_first = True
                sl_bar = entry_idx + 1 + i
                break
            if l <= take_profit:
                hit_tp_first = True
                tp_bar = entry_idx + 1 + i
                break
        
        # 計算結果
        if hit_tp_first:
            result = 'win'
            exit_price = take_profit
            profit = entry_price - take_profit
            profit_pct = (profit / entry_price) * 100
            exit_bar = tp_bar
        elif hit_sl_first:
            result = 'loss'
            exit_price = stop_loss
            profit = entry_price - stop_loss
            profit_pct = (profit / entry_price) * 100
            exit_bar = sl_bar
        else:
            result = 'no_exit'
            exit_price = self.df.iloc[entry_idx + lookforward_bars]['Close']
            profit = entry_price - exit_price
            profit_pct = (profit / entry_price) * 100
            exit_bar = entry_idx + lookforward_bars
        
        trade_info = {
            'entry_idx': entry_idx,
            'entry_price': entry_price,
            'entry_candle': entry_candle.to_dict(),
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'atr_value': atr_value,
            'direction': 'short',
            'result': result,
            'exit_price': exit_price,
            'exit_bar': exit_bar,
            'profit': profit,
            'profit_pct': profit_pct,
            'bars_held': (exit_bar - entry_idx) if exit_bar else lookforward_bars
        }
        
        return trade_info
    
    def test_all_longs(self, lookforward_bars: int = 50):
        """
        在每根K棒都試著開多單，看結果如何。
        """
        for i in range(len(self.df) - lookforward_bars):
            trade = self.backtest_long_from_candle(i, lookforward_bars)
            if trade:
                self.trades.append(trade)
                
                if trade['result'] == 'win':
                    self.winning_trades.append(trade)
                elif trade['result'] == 'loss':
                    self.losing_trades.append(trade)
    
    def test_all_shorts(self, lookforward_bars: int = 50):
        """
        在每根K棒都試著開空單，看結果如何。
        """
        for i in range(len(self.df) - lookforward_bars):
            trade = self.backtest_short_from_candle(i, lookforward_bars)
            if trade:
                self.trades.append(trade)
                
                if trade['result'] == 'win':
                    self.winning_trades.append(trade)
                elif trade['result'] == 'loss':
                    self.losing_trades.append(trade)
    
    def get_winning_entry_indices(self) -> List[int]:
        """
        獲取所有獲利交易的入場K棒索引。
        """
        return [trade['entry_idx'] for trade in self.winning_trades]
    
    def get_losing_entry_indices(self) -> List[int]:
        """
        獲取所有虧損交易的入場K棒索引。
        """
        return [trade['entry_idx'] for trade in self.losing_trades]
    
    def get_statistics(self) -> Dict:
        """
        獲取統計信息。
        """
        if not self.trades:
            return {}
        
        total_trades = len(self.trades)
        wins = len(self.winning_trades)
        losses = len(self.losing_trades)
        no_exit = total_trades - wins - losses
        
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        avg_win = np.mean([t['profit_pct'] for t in self.winning_trades]) if self.winning_trades else 0
        avg_loss = np.mean([t['profit_pct'] for t in self.losing_trades]) if self.losing_trades else 0
        
        total_profit = sum([t['profit'] for t in self.trades])
        
        return {
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'no_exit': no_exit,
            'win_rate': win_rate,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'total_profit': total_profit,
            'profit_factor': abs(sum([t['profit'] for t in self.winning_trades]) / sum([t['profit'] for t in self.losing_trades])) if self.losing_trades else 0
        }
    
    def save_winning_reversals(self, output_file: str):
        """
        保存獲利交易的K棒索引到JSON。
        """
        indices = self.get_winning_entry_indices()
        
        data = {
            'method': 'practical_reversal_trading',
            'setup': 'Long or Short, SL=1*ATR, TP=1.5*ATR',
            'reversal_indices': indices,
            'total_reversals': len(indices),
            'total_candles': len(self.df),
            'statistics': self.get_statistics()
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"Saved {len(indices)} winning reversal points to {output_file}")


def main():
    """CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Find reversal points based on practical trading outcomes'
    )
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading pair')
    parser.add_argument('--lookback', type=int, default=500, help='Number of candles')
    parser.add_argument('--lookforward', type=int, default=50, 
                       help='How many bars forward to check for SL/TP hit')
    parser.add_argument('--direction', type=str, default='both',
                       choices=['long', 'short', 'both'],
                       help='Trade direction to test')
    parser.add_argument('--output', type=str, help='Save reversal indices to JSON')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading {args.symbol} data...")
    try:
        from huggingface_hub import hf_hub_download
        
        REPO_ID = "zongowo111/v2-crypto-ohlcv-data"
        base = args.symbol.replace("USDT", "")
        path_in_repo = f"klines/{args.symbol}/{base}_15m.parquet"
        
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=path_in_repo,
            repo_type="dataset"
        )
        df = pd.read_parquet(local_path)
        df.columns = [col.lower() for col in df.columns]
        df = df[['open', 'high', 'low', 'close', 'volume']].copy()
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df.tail(args.lookback).reset_index(drop=True)
        
        print(f"Loaded {len(df)} candles\n")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Test trades
    print(f"Testing trades (SL=1*ATR, TP=1.5*ATR, lookforward={args.lookforward} bars)...\n")
    tester = PracticalReversalTester(df)
    
    if args.direction in ['long', 'both']:
        print(f"Testing LONG trades...")
        tester.test_all_longs(lookforward_bars=args.lookforward)
    
    if args.direction in ['short', 'both']:
        print(f"Testing SHORT trades...")
        tester.test_all_shorts(lookforward_bars=args.lookforward)
    
    # Get statistics
    stats = tester.get_statistics()
    
    print("\nTrade Statistics:")
    print("=" * 70)
    print(f"Total Trades:        {stats['total_trades']}")
    print(f"Winning Trades:      {stats['wins']} ({stats['win_rate']:.1f}%)")
    print(f"Losing Trades:       {stats['losses']}")
    print(f"No Exit:             {stats['no_exit']}")
    print(f"\nAverage Win:         {stats['avg_win_pct']:.3f}%")
    print(f"Average Loss:        {stats['avg_loss_pct']:.3f}%")
    print(f"Profit Factor:       {stats['profit_factor']:.2f}")
    print(f"Total Profit:        {stats['total_profit']:.2f}")
    
    # Get reversal indices
    winning_indices = tester.get_winning_entry_indices()
    losing_indices = tester.get_losing_entry_indices()
    
    print(f"\n{'=' * 70}")
    print(f"\nWinning Reversal Points: {len(winning_indices)}")
    print(f"First 20 winning entries: {winning_indices[:20]}")
    
    print(f"\nLosing Entry Points: {len(losing_indices)}")
    print(f"First 20 losing entries: {losing_indices[:20]}")
    
    # Save if requested
    if args.output:
        tester.save_winning_reversals(args.output)


if __name__ == "__main__":
    main()
