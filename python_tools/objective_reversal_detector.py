#!/usr/bin/env python3

"""
Objective Reversal Detector - Automatically identify reversal points

Methods:
1. Local Extrema Detection: Find local highs/lows
2. Trend Reversal Detection: Detect trend direction changes
3. Support/Resistance Bounce: Detect bounces off key levels
4. Pattern Recognition: Identify reversal candlestick patterns
5. Momentum Reversal: Detect RSI/MACD extremes
6. Volatility Clustering: Identify low volatility before breakout
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
import warnings
warnings.filterwarnings('ignore')


class ObjectiveReversalDetector:
    """Detect reversals using objective, quantifiable methods."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize detector.
        
        Args:
            df: DataFrame with OHLCV data
        """
        self.df = df.copy()
        self.reversals = {}
        self._prepare_indicators()
    
    def _prepare_indicators(self):
        """Calculate all indicators."""
        df = self.df
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['Close'], 14)
        
        # MACD
        ema12 = df['Close'].ewm(span=12).mean()
        ema26 = df['Close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # ATR
        df['atr'] = self._calculate_atr(df, 14)
        
        # EMA
        df['ema20'] = df['Close'].ewm(span=20).mean()
        df['ema50'] = df['Close'].ewm(span=50).mean()
        
        # Price velocity
        df['velocity'] = df['Close'].diff()
        df['acceleration'] = df['velocity'].diff()
        
        self.df = df
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 0.001)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR."""
        tr1 = df['High'] - df['Low']
        tr2 = abs(df['High'] - df['Close'].shift())
        tr3 = abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr
    
    def detect_local_extrema(self, window: int = 5) -> np.ndarray:
        """
        方法1：檢測局部極值（高點和低點）
        
        當價格形成局部最高點或最低點時，常常是反轉點。
        
        Args:
            window: 檢查左右多少根蠟燭
        
        Returns:
            反轉點數組 (1 = 反轉, 0 = 非反轉)
        """
        reversals = np.zeros(len(self.df))
        closes = self.df['Close'].values
        
        for i in range(window, len(closes) - window):
            current = closes[i]
            
            # 檢查是否是局部最高點
            is_local_high = (
                current >= np.max(closes[i-window:i]) and
                current > np.max(closes[i+1:i+window+1])
            )
            
            # 檢查是否是局部最低點
            is_local_low = (
                current <= np.min(closes[i-window:i]) and
                current < np.min(closes[i+1:i+window+1])
            )
            
            if is_local_high or is_local_low:
                reversals[i] = 1
        
        self.reversals['local_extrema'] = reversals
        return reversals
    
    def detect_trend_reversal(self, ema_short: int = 9, ema_long: int = 21) -> np.ndarray:
        """
        方法2：檢測趨勢反轉
        
        當快速EMA穿過慢速EMA時，趨勢反轉。
        
        Args:
            ema_short: 短期EMA週期
            ema_long: 長期EMA週期
        
        Returns:
            反轉點數組
        """
        reversals = np.zeros(len(self.df))
        
        ema_s = self.df['Close'].ewm(span=ema_short).mean()
        ema_l = self.df['Close'].ewm(span=ema_long).mean()
        
        for i in range(1, len(self.df)):
            # 黃金交叉（死亡交叉）
            prev_above = ema_s.iloc[i-1] > ema_l.iloc[i-1]
            curr_above = ema_s.iloc[i] > ema_l.iloc[i]
            
            if prev_above != curr_above:
                reversals[i] = 1
        
        self.reversals['trend_reversal'] = reversals
        return reversals
    
    def detect_rsi_extremes(self, oversold: int = 30, overbought: int = 70) -> np.ndarray:
        """
        方法3：檢測RSI極值
        
        RSI低於30或高於70通常預示反轉。
        
        Args:
            oversold: 超賣閾值
            overbought: 超買閾值
        
        Returns:
            反轉點數組
        """
        reversals = np.zeros(len(self.df))
        rsi = self.df['rsi'].values
        
        for i in range(1, len(rsi)):
            prev_rsi = rsi[i-1]
            curr_rsi = rsi[i]
            
            # 從超賣反彈
            if prev_rsi < oversold and curr_rsi > oversold:
                reversals[i] = 1
            
            # 從超買回落
            if prev_rsi > overbought and curr_rsi < overbought:
                reversals[i] = 1
        
        self.reversals['rsi_extremes'] = reversals
        return reversals
    
    def detect_macd_crossover(self) -> np.ndarray:
        """
        方法4：檢測MACD交叉
        
        MACD線與信號線的交叉點通常是反轉點。
        
        Returns:
            反轉點數組
        """
        reversals = np.zeros(len(self.df))
        macd = self.df['macd'].values
        signal = self.df['macd_signal'].values
        
        for i in range(1, len(macd)):
            prev_above = macd[i-1] > signal[i-1]
            curr_above = macd[i] > signal[i]
            
            if prev_above != curr_above:
                reversals[i] = 1
        
        self.reversals['macd_crossover'] = reversals
        return reversals
    
    def detect_candlestick_patterns(self) -> np.ndarray:
        """
        方法5：檢測反轉型K線型態
        
        檢測：錘子線、上吊線、射擊之星、倒錘子線
        
        Returns:
            反轉點數組
        """
        reversals = np.zeros(len(self.df))
        
        o = self.df['Open'].values
        h = self.df['High'].values
        l = self.df['Low'].values
        c = self.df['Close'].values
        
        for i in range(len(self.df)):
            range_hl = h[i] - l[i]
            if range_hl == 0:
                continue
            
            body = abs(c[i] - o[i])
            upper_wick = h[i] - max(o[i], c[i])
            lower_wick = min(o[i], c[i]) - l[i]
            
            # 錘子線：下影線長，上影線短，實體小，在低位
            is_hammer = (
                lower_wick > range_hl * 0.5 and
                upper_wick < range_hl * 0.1 and
                body < range_hl * 0.3
            )
            
            # 射擊之星：上影線長，下影線短，實體小，在高位
            is_shooting_star = (
                upper_wick > range_hl * 0.5 and
                lower_wick < range_hl * 0.1 and
                body < range_hl * 0.3
            )
            
            # 上吊線：類似錘子但在高位
            is_hanging_man = (
                lower_wick > range_hl * 0.5 and
                upper_wick < range_hl * 0.1 and
                body < range_hl * 0.3 and
                c[i] < o[i]  # 紅棒
            )
            
            # 倒錘子線：上影線長，下影線短，在低位
            is_inverted_hammer = (
                upper_wick > range_hl * 0.5 and
                lower_wick < range_hl * 0.1 and
                body < range_hl * 0.3 and
                c[i] > o[i]  # 綠棒
            )
            
            if is_hammer or is_shooting_star or is_hanging_man or is_inverted_hammer:
                reversals[i] = 1
        
        self.reversals['candlestick_patterns'] = reversals
        return reversals
    
    def detect_support_resistance_bounce(self, window: int = 20) -> np.ndarray:
        """
        方法6：檢測支撐/阻力反彈
        
        價格接近先前的高低點並反彈。
        
        Args:
            window: 檢查多少根蠟燭內的先前極值
        
        Returns:
            反轉點數組
        """
        reversals = np.zeros(len(self.df))
        
        closes = self.df['Close'].values
        
        for i in range(window, len(closes)):
            current = closes[i]
            
            # 尋找過去window根蠟燭中的高點和低點
            past_high = np.max(closes[i-window:i])
            past_low = np.min(closes[i-window:i])
            
            range_price = past_high - past_low
            
            # 如果接近過去的高點或低點（在2%以內）且反向
            close_to_high = abs(current - past_high) < range_price * 0.02
            close_to_low = abs(current - past_low) < range_price * 0.02
            
            if close_to_high or close_to_low:
                # 檢查下一根蠟燭是否反向
                if i + 1 < len(closes):
                    next_close = closes[i+1]
                    if close_to_high and next_close < current:
                        reversals[i] = 1
                    if close_to_low and next_close > current:
                        reversals[i] = 1
        
        self.reversals['support_resistance'] = reversals
        return reversals
    
    def detect_volatility_expansion(self, lookback: int = 20, threshold: float = 1.5) -> np.ndarray:
        """
        方法7：檢測波動率擴張
        
        低波動率之後的波動率擴張通常預示反轉。
        
        Args:
            lookback: 檢查過去多少根蠟燭
            threshold: 倍數閾值
        
        Returns:
            反轉點數組
        """
        reversals = np.zeros(len(self.df))
        
        atr = self.df['atr'].values
        
        for i in range(lookback, len(atr)):
            past_atr = np.mean(atr[i-lookback:i])
            current_atr = atr[i]
            
            # ATR突然擴張（波動率增加）
            if current_atr > past_atr * threshold:
                reversals[i] = 1
        
        self.reversals['volatility_expansion'] = reversals
        return reversals
    
    def detect_all(self) -> Dict[str, np.ndarray]:
        """
        執行所有檢測方法。
        
        Returns:
            包含所有檢測結果的字典
        """
        self.detect_local_extrema()
        self.detect_trend_reversal()
        self.detect_rsi_extremes()
        self.detect_macd_crossover()
        self.detect_candlestick_patterns()
        self.detect_support_resistance_bounce()
        self.detect_volatility_expansion()
        
        return self.reversals
    
    def ensemble_detection(self, min_agreements: int = 2) -> np.ndarray:
        """
        集合投票：只有當多個方法都同意時才標記為反轉點。
        
        Args:
            min_agreements: 需要多少個方法同意
        
        Returns:
            最終反轉點數組
        """
        if not self.reversals:
            self.detect_all()
        
        # 疊加所有檢測結果
        combined = np.zeros(len(self.df))
        
        for method_result in self.reversals.values():
            combined += method_result
        
        # 只保留達到最低同意數的
        final = (combined >= min_agreements).astype(int)
        
        return final
    
    def get_reversal_indices(self, method: str = 'ensemble', min_agreements: int = 2) -> List[int]:
        """
        獲取反轉點的索引列表。
        
        Args:
            method: 'ensemble' 或特定方法名稱
            min_agreements: 集合投票最低同意數
        
        Returns:
            反轉點索引列表
        """
        if method == 'ensemble':
            reversals = self.ensemble_detection(min_agreements)
        else:
            if method not in self.reversals:
                self.detect_all()
            reversals = self.reversals.get(method, np.zeros(len(self.df)))
        
        return np.where(reversals == 1)[0].tolist()
    
    def get_statistics(self) -> Dict:
        """
        獲取檢測統計信息。
        
        Returns:
            統計字典
        """
        if not self.reversals:
            self.detect_all()
        
        stats = {}
        total_candles = len(self.df)
        
        for method_name, reversals_array in self.reversals.items():
            count = int(np.sum(reversals_array))
            percentage = 100 * count / total_candles
            stats[method_name] = {
                'count': count,
                'percentage': percentage
            }
        
        ensemble = self.ensemble_detection(min_agreements=2)
        stats['ensemble_2agreements'] = {
            'count': int(np.sum(ensemble)),
            'percentage': 100 * np.sum(ensemble) / total_candles
        }
        
        ensemble3 = self.ensemble_detection(min_agreements=3)
        stats['ensemble_3agreements'] = {
            'count': int(np.sum(ensemble3)),
            'percentage': 100 * np.sum(ensemble3) / total_candles
        }
        
        return stats


def main():
    """CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Objectively detect reversal points in OHLCV data'
    )
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading pair')
    parser.add_argument('--lookback', type=int, default=500, help='Number of candles')
    parser.add_argument('--method', type=str, default='ensemble',
                       help='Detection method: ensemble, local_extrema, trend_reversal, rsi_extremes, macd_crossover, candlestick_patterns, support_resistance, volatility_expansion')
    parser.add_argument('--min-agreements', type=int, default=2,
                       help='Minimum agreements for ensemble method')
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
    
    # Detect reversals
    print("Detecting reversal points...\n")
    detector = ObjectiveReversalDetector(df)
    
    if args.method == 'ensemble':
        reversals = detector.ensemble_detection(min_agreements=args.min_agreements)
        print(f"Ensemble Detection (min {args.min_agreements} agreements):\n")
    else:
        if args.method == 'local_extrema':
            reversals = detector.detect_local_extrema()
        elif args.method == 'trend_reversal':
            reversals = detector.detect_trend_reversal()
        elif args.method == 'rsi_extremes':
            reversals = detector.detect_rsi_extremes()
        elif args.method == 'macd_crossover':
            reversals = detector.detect_macd_crossover()
        elif args.method == 'candlestick_patterns':
            reversals = detector.detect_candlestick_patterns()
        elif args.method == 'support_resistance':
            reversals = detector.detect_support_resistance_bounce()
        elif args.method == 'volatility_expansion':
            reversals = detector.detect_volatility_expansion()
        else:
            reversals = detector.ensemble_detection()
        
        print(f"Detection Method: {args.method}\n")
    
    # Get statistics
    stats = detector.get_statistics()
    
    print("Detection Statistics:")
    print("=" * 70)
    for method_name, method_stats in stats.items():
        print(f"{method_name:30s}: {method_stats['count']:4d} ({method_stats['percentage']:5.2f}%)")
    
    # Get indices
    indices = detector.get_reversal_indices(method=args.method, min_agreements=args.min_agreements)
    
    print(f"\n{'=' * 70}")
    print(f"\nFound {len(indices)} reversal points")
    print(f"\nFirst 20 reversal indices:")
    print(indices[:20])
    
    # Save if requested
    if args.output:
        output_data = {
            'method': args.method,
            'reversal_indices': indices,
            'total_reversals': len(indices),
            'total_candles': len(df),
            'reversal_percentage': 100 * len(indices) / len(df)
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nReversal indices saved to: {args.output}")


if __name__ == "__main__":
    main()
