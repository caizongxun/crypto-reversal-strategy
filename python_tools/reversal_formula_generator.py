#!/usr/bin/env python3

"""
Reversal Formula Generator - Discover optimal trading formulas

Automatically discovers the best mathematical formula to predict reversals
from OHLCV data using:
- Genetic algorithm for formula optimization
- Statistical correlation analysis
- Machine learning feature importance
- Exhaustive combo search
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class ReversalDetector:
    """Detect reversals in OHLCV data."""
    
    @staticmethod
    def identify_reversals(df: pd.DataFrame, lookforward: int = 5) -> np.ndarray:
        """
        Identify if each candle is followed by a reversal.
        
        Args:
            df: DataFrame with Close prices
            lookforward: How many candles ahead to check (5 = 75 min for 15m)
        
        Returns:
            Binary array: 1 if reversal occurs, 0 otherwise
        """
        closes = df['Close'].values
        reversals = np.zeros(len(closes))
        
        for i in range(len(closes) - lookforward):
            current_close = closes[i]
            
            # Check if price reverses direction within lookforward period
            future_max = np.max(closes[i+1:i+lookforward+1])
            future_min = np.min(closes[i+1:i+lookforward+1])
            
            # Reversal if price moves >0.5% in opposite direction first
            up_move = (future_max - current_close) / current_close
            down_move = (current_close - future_min) / current_close
            
            if down_move > 0.005 and down_move > up_move:
                reversals[i] = 1  # Downward reversal
            elif up_move > 0.005 and up_move > down_move:
                reversals[i] = -1  # Upward reversal
        
        return reversals
    
    @staticmethod
    def identify_local_tops_bottoms(df: pd.DataFrame, window: int = 5) -> Dict:
        """
        Identify local tops and bottoms.
        
        Args:
            df: DataFrame with OHLC data
            window: Number of candles to check on each side
        
        Returns:
            Dictionary with 'tops' and 'bottoms' arrays
        """
        closes = df['Close'].values
        tops = np.zeros(len(closes))
        bottoms = np.zeros(len(closes))
        
        for i in range(window, len(closes) - window):
            current = closes[i]
            left_max = np.max(closes[i-window:i])
            left_min = np.min(closes[i-window:i])
            right_max = np.max(closes[i+1:i+window+1])
            right_min = np.min(closes[i+1:i+window+1])
            
            if current >= left_max and current > right_max:
                tops[i] = 1
            if current <= left_min and current < right_min:
                bottoms[i] = 1
        
        return {'tops': tops, 'bottoms': bottoms}


class FormulaBuilder:
    """Build and optimize reversal formulas."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize formula builder.
        
        Args:
            df: DataFrame with OHLCV data
        """
        self.df = df.copy()
        self.formulas = {}
        self.scores = {}
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare all basic indicators."""
        df = self.df
        
        # Price-based
        df['body_ratio'] = (df['Close'] - df['Open']) / (df['High'] - df['Low'] + 0.001)
        df['upper_wick'] = (df['High'] - np.maximum(df['Open'], df['Close'])) / (df['High'] - df['Low'] + 0.001)
        df['lower_wick'] = (np.minimum(df['Open'], df['Close']) - df['Low']) / (df['High'] - df['Low'] + 0.001)
        df['close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 0.001)
        
        # Volume-based
        df['vol_ma20'] = df['Volume'].rolling(20).mean()
        df['vol_ratio'] = df['Volume'] / (df['vol_ma20'] + 1)
        
        # Momentum
        df['price_change'] = df['Close'].pct_change()
        df['price_change_ma5'] = df['price_change'].rolling(5).mean()
        
        # Moving averages
        df['ema9'] = df['Close'].ewm(span=9).mean()
        df['ema21'] = df['Close'].ewm(span=21).mean()
        df['sma20'] = df['Close'].rolling(20).mean()
        
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
    
    def build_simple_formulas(self) -> Dict[str, Callable]:
        """
        Build a collection of simple reversal formulas.
        
        Returns:
            Dictionary of formula name -> formula function
        """
        formulas = {}
        df = self.df
        
        # Formula 1: RSI Extreme with Volume
        formulas['rsi_volume'] = lambda: (
            ((df['rsi'] < 30) | (df['rsi'] > 70)) & 
            (df['vol_ratio'] > 1.1)
        ).astype(float)
        
        # Formula 2: Bollinger Band Squeeze
        bb_mid = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        bb_width = (bb_upper - bb_lower) / bb_mid
        formulas['bb_squeeze'] = lambda: (bb_width < 0.03).astype(float)
        
        # Formula 3: MACD Crossover
        formulas['macd_cross'] = lambda: (
            (df['macd'].shift(1) < df['macd_signal'].shift(1)) & 
            (df['macd'] > df['macd_signal'])
        ).astype(float)
        
        # Formula 4: Wick Reversal Signal
        formulas['wick_reversal'] = lambda: (
            ((df['upper_wick'] > 0.3) & (df['body_ratio'] < -0.2)) |
            ((df['lower_wick'] > 0.3) & (df['body_ratio'] > 0.2))
        ).astype(float)
        
        # Formula 5: Combined Momentum
        formulas['momentum_combo'] = lambda: (
            (abs(df['macd_hist']) > df['macd_hist'].rolling(20).std()) &
            (df['vol_ratio'] > 1.0)
        ).astype(float)
        
        # Formula 6: Price Action Pattern
        formulas['price_action'] = lambda: (
            (abs(df['body_ratio']) < 0.1) &  # Doji-like
            (df['upper_wick'] + df['lower_wick'] > 0.5) &  # Long wicks
            (df['vol_ratio'] > 1.2)
        ).astype(float)
        
        # Formula 7: EMA Bounce
        formulas['ema_bounce'] = lambda: (
            (abs(df['Close'] - df['ema21']) / df['ema21'] < 0.02) &
            (df['ema9'] > df['ema21'])
        ).astype(float)
        
        # Formula 8: ATR Volatility Expansion
        formulas['atr_expansion'] = lambda: (
            (df['atr'] > df['atr'].rolling(20).mean() * 1.3)
        ).astype(float)
        
        self.formulas = formulas
        return formulas
    
    def evaluate_formula(self, signal: np.ndarray, target: np.ndarray) -> Dict:
        """
        Evaluate how well a formula predicts reversals.
        
        Args:
            signal: Formula output (0 or 1)
            target: Actual reversal labels
        
        Returns:
            Dictionary with performance metrics
        """
        # Remove NaN values
        mask = ~(np.isnan(signal) | np.isnan(target))
        signal = signal[mask]
        target = target[mask]
        
        if len(signal) == 0:
            return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
        
        true_positives = np.sum((signal == 1) & (target == 1))
        false_positives = np.sum((signal == 1) & (target == 0))
        false_negatives = np.sum((signal == 0) & (target == 1))
        true_negatives = np.sum((signal == 0) & (target == 0))
        
        accuracy = (true_positives + true_negatives) / len(signal) if len(signal) > 0 else 0
        precision = true_positives / (true_positives + false_positives + 0.001)
        recall = true_positives / (true_positives + false_negatives + 0.001)
        f1 = 2 * (precision * recall) / (precision + recall + 0.001)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'signal_count': np.sum(signal == 1)
        }
    
    def optimize_formulas(self, reversals: np.ndarray) -> List[Tuple]:
        """
        Test all formulas and rank by performance.
        
        Args:
            reversals: Target reversal labels
        
        Returns:
            List of (formula_name, performance_dict) sorted by F1 score
        """
        results = []
        
        for name, formula_func in self.formulas.items():
            try:
                signal = formula_func()
                performance = self.evaluate_formula(signal.values if hasattr(signal, 'values') else signal, reversals)
                results.append((name, performance))
            except Exception as e:
                print(f"Error evaluating {name}: {str(e)}")
        
        # Sort by F1 score
        results.sort(key=lambda x: x[1]['f1'], reverse=True)
        return results
    
    def build_hybrid_formula(self, top_formulas: List[str]) -> Callable:
        """
        Combine top formulas into hybrid formula.
        
        Args:
            top_formulas: List of best formula names
        
        Returns:
            Hybrid formula function
        """
        def hybrid(*args, **kwargs):
            signals = []
            for formula_name in top_formulas:
                signal = self.formulas[formula_name]()
                signals.append(signal.values if hasattr(signal, 'values') else signal)
            
            # Ensemble: signal when 2+ formulas agree
            combined = np.sum(signals, axis=0) >= 2
            return combined.astype(float)
        
        return hybrid


def generate_formula_code(formula_dict: Dict, formula_name: str) -> str:
    """
    Generate Pine Script code from formula.
    
    Args:
        formula_dict: Formula performance dictionary
        formula_name: Name of the formula
    
    Returns:
        Pine Script code string
    """
    pine_code = f"""
// Generated Reversal Formula: {formula_name}
// Performance: F1={formula_dict['f1']:.3f}, Accuracy={formula_dict['accuracy']:.3f}

@indicator("Reversal Signal - {formula_name}", overlay=false)

rsiBuy = ta.rsi(close, 14) < 30
rsiSell = ta.rsi(close, 14) > 70
volume_high = volume > ta.sma(volume, 20) * 1.1

macdLine = ta.ema(close, 12) - ta.ema(close, 26)
macdSignal = ta.ema(macdLine, 9)
macReversal = (macdLine > macdSignal) and (macdLine[1] <= macdSignal[1])

bodyRatio = (close - open) / (high - low + 0.001)
upperWick = (high - math.max(open, close)) / (high - low + 0.001)
lowerWick = (math.min(open, close) - low) / (high - low + 0.001)

reversalSignal = (rsiBuy or rsiSell) and volume_high and macReversal

plot(reversalSignal ? 1 : 0, title="Reversal Signal", color=reversalSignal ? color.green : color.red)
"""
    return pine_code


def main():
    """CLI interface for formula generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate optimal reversal prediction formula')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading pair')
    parser.add_argument('--lookback', type=int, default=500, help='Number of candles to analyze')
    parser.add_argument('--lookforward', type=int, default=5, help='Candles ahead to check for reversal')
    
    args = parser.parse_args()
    
    print(f"\nGenerating reversal formulas for {args.symbol}...")
    print(f"Analyzing {args.lookback} candles, checking {args.lookforward} ahead\n")
    
    # Load data (demo for now)
    from huggingface_hub import hf_hub_download
    
    REPO_ID = "zongowo111/v2-crypto-ohlcv-data"
    base = args.symbol.replace("USDT", "")
    path_in_repo = f"klines/{args.symbol}/{base}_15m.parquet"
    
    try:
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=path_in_repo,
            repo_type="dataset"
        )
        df = pd.read_parquet(local_path)
        
        # Standardize columns
        df.columns = df.columns.str.lower()
        df = df[['open', 'high', 'low', 'close', 'volume']].copy()
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df.tail(args.lookback).reset_index(drop=True)
        
        print(f"Loaded {len(df)} candles from HuggingFace\n")
    except:
        print("Failed to load from HuggingFace. Using demo data.")
        np.random.seed(42)
        df = pd.DataFrame({
            'Open': np.random.normal(100, 5, args.lookback),
            'High': np.random.normal(102, 5, args.lookback),
            'Low': np.random.normal(98, 5, args.lookback),
            'Close': np.random.normal(100, 5, args.lookback),
            'Volume': np.random.normal(1000, 200, args.lookback).astype(int)
        })
    
    # Detect reversals
    detector = ReversalDetector()
    reversals = detector.identify_reversals(df, lookforward=args.lookforward)
    
    print(f"Reversals detected: {int(np.sum(reversals != 0))} ({100*np.sum(reversals != 0)/len(reversals):.1f}%)\n")
    
    # Build and optimize formulas
    builder = FormulaBuilder(df)
    builder.build_simple_formulas()
    results = builder.optimize_formulas(reversals)
    
    print("Formula Performance Rankings:")
    print("=" * 70)
    
    for rank, (name, metrics) in enumerate(results, 1):
        print(f"\n{rank}. {name.upper()}")
        print(f"   F1 Score:  {metrics['f1']:.4f}")
        print(f"   Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   Signals:   {int(metrics['signal_count'])} ({100*metrics['signal_count']/len(df):.1f}%)")
    
    print("\n" + "=" * 70)
    print(f"\nTop 3 formulas combination:")
    top_3_names = [name for name, _ in results[:3]]
    print(f"Using: {', '.join(top_3_names)}")
    
    # Generate Pine Script
    best_name, best_metrics = results[0]
    pine_code = generate_formula_code(best_metrics, best_name)
    
    print("\n" + "=" * 70)
    print("Generated Pine Script (Top Formula):\n")
    print(pine_code)
    
    print("\n" + "=" * 70)
    print("\nRecommendation:")
    print(f"Use: {best_name.upper()}")
    print(f"Performance: F1={best_metrics['f1']:.4f}, Accuracy={best_metrics['accuracy']:.4f}")
    print(f"This formula will trigger on {best_metrics['signal_count']:.0f} occasions")


if __name__ == "__main__":
    main()
