#!/usr/bin/env python3

"""
Feature Generator - Automated OHLCV feature discovery

This tool generates new features from historical OHLCV data using:
- Combinatorial feature mining
- Temporal feature engineering
- K-line pattern quantification
- Volume profile extraction

Supports data loading from:
- Local CSV files
- HuggingFace datasets (zongowo111/v2-crypto-ohlcv-data)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import argparse
import warnings
warnings.filterwarnings('ignore')

try:
    from huggingface_hub import hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class HuggingFaceDataLoader:
    """Load OHLCV data from HuggingFace datasets."""
    
    REPO_ID = "zongowo111/v2-crypto-ohlcv-data"
    HF_ROOT = "klines"
    
    SUPPORTED_SYMBOLS = [
        'AAVEUSDT', 'ADAUSDT', 'ALGOUSDT', 'ARBUSDT', 'ATOMUSDT', 'AVAXUSDT',
        'BALUSDT', 'BATUSDT', 'BCHUSDT', 'BNBUSDT', 'BTCUSDT', 'COMPUSDT',
        'CRVUSDT', 'DOGEUSDT', 'DOTUSDT', 'ENJUSDT', 'ENSUSDT', 'ETCUSDT',
        'ETHUSDT', 'FILUSDT', 'GALAUSDT', 'GRTUSDT', 'IMXUSDT', 'KAVAUSDT',
        'LINKUSDT', 'LTCUSDT', 'MANAUSDT', 'MATICUSDT', 'MKRUSDT', 'NEARUSDT',
        'OPUSDT', 'SANDUSDT', 'SNXUSDT', 'SOLUSDT', 'SPELLUSDT', 'UNIUSDT',
        'XRPUSDT', 'ZRXUSDT'
    ]
    
    @staticmethod
    def validate_symbol(symbol: str) -> bool:
        """Check if symbol is supported."""
        return symbol.upper() in HuggingFaceDataLoader.SUPPORTED_SYMBOLS
    
    @staticmethod
    def load_klines(symbol: str, timeframe: str = '15m') -> pd.DataFrame:
        """
        Load OHLCV data from HuggingFace dataset.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            timeframe: '15m' or '1h'
        
        Returns:
            DataFrame with columns: open_time, open, high, low, close, volume, etc.
        
        Raises:
            ValueError: If symbol not supported or timeframe invalid
            ImportError: If huggingface_hub not installed
        """
        if not HF_AVAILABLE:
            raise ImportError(
                "huggingface_hub not installed. "
                "Install with: pip install huggingface_hub"
            )
        
        symbol = symbol.upper()
        if not HuggingFaceDataLoader.validate_symbol(symbol):
            raise ValueError(
                f"Symbol {symbol} not supported. "
                f"Supported: {HuggingFaceDataLoader.SUPPORTED_SYMBOLS}"
            )
        
        if timeframe not in ['15m', '1h']:
            raise ValueError("Timeframe must be '15m' or '1h'")
        
        base = symbol.replace("USDT", "")
        filename = f"{base}_{timeframe}.parquet"
        path_in_repo = f"{HuggingFaceDataLoader.HF_ROOT}/{symbol}/{filename}"
        
        print(f"Loading {symbol} {timeframe} from HuggingFace...")
        
        try:
            local_path = hf_hub_download(
                repo_id=HuggingFaceDataLoader.REPO_ID,
                filename=path_in_repo,
                repo_type="dataset"
            )
            df = pd.read_parquet(local_path)
            print(f"Successfully loaded {len(df)} candles")
            return df
        except Exception as e:
            raise RuntimeError(f"Failed to load data from HuggingFace: {str(e)}")
    
    @staticmethod
    def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names to expected format.
        
        HuggingFace data has columns: open_time, open, high, low, close, volume, etc.
        We rename to: Open, High, Low, Close, Volume (capitalized)
        """
        column_mapping = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
            'open_time': 'Timestamp'
        }
        
        df_copy = df.copy()
        df_copy = df_copy.rename(columns=column_mapping)
        
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df_copy.columns for col in required_cols):
            raise ValueError(
                f"DataFrame missing required columns. "
                f"Expected: {required_cols}, Got: {df_copy.columns.tolist()}"
            )
        
        return df_copy[required_cols + ['Timestamp']]


class FeatureGenerator:
    """Generate new features from OHLCV data."""
    
    def __init__(self, df: pd.DataFrame, lookback: int = 100):
        """
        Initialize feature generator.
        
        Args:
            df: DataFrame with columns [Open, High, Low, Close, Volume]
            lookback: Number of historical periods to use
        """
        self.df = df.tail(lookback).reset_index(drop=True)
        self.lookback = lookback
        self.features = {}
    
    def generate_combo_features(self) -> Dict:
        """
        Generate combinatorial features from OHLC.
        
        Returns:
            Dictionary of combo features
        """
        combo = {}
        
        for i in range(1, len(self.df)):
            o = self.df.loc[i, 'Open']
            h = self.df.loc[i, 'High']
            l = self.df.loc[i, 'Low']
            c = self.df.loc[i, 'Close']
            v = self.df.loc[i, 'Volume']
            
            range_hl = h - l
            if range_hl == 0:
                range_hl = 0.0001
            
            combo[f'body_ratio_{i}'] = (c - o) / range_hl
            combo[f'upper_wick_{i}'] = (h - max(o, c)) / range_hl
            combo[f'lower_wick_{i}'] = (min(o, c) - l) / range_hl
            combo[f'close_change_{i}'] = (c - self.df.loc[i-1, 'Close']) / self.df.loc[i-1, 'Close']
            combo[f'vol_ratio_{i}'] = v / self.df['Volume'].tail(20).mean()
        
        self.features['combo'] = combo
        return combo
    
    def generate_candlestick_features(self) -> Dict:
        """
        Quantify candlestick patterns.
        
        Returns:
            Dictionary of pattern features
        """
        patterns = {}
        
        for i in range(2, len(self.df)):
            o0, h0, l0, c0 = self.df.loc[i, ['Open', 'High', 'Low', 'Close']]
            o1, h1, l1, c1 = self.df.loc[i-1, ['Open', 'High', 'Low', 'Close']]
            o2, h2, l2, c2 = self.df.loc[i-2, ['Open', 'High', 'Low', 'Close']]
            
            range_hl = h0 - l0
            if range_hl == 0:
                range_hl = 0.0001
            
            patterns[f'higher_high_{i}'] = 1 if h0 > h1 and h1 > h2 else 0
            patterns[f'lower_low_{i}'] = 1 if l0 < l1 and l1 < l2 else 0
            patterns[f'reverse_signal_{i}'] = 1 if (h0 > h1 and l0 < l1) else 0
            patterns[f'body_size_{i}'] = abs(c0 - o0) / range_hl
            patterns[f'close_position_{i}'] = (c0 - l0) / range_hl if range_hl > 0 else 0.5
        
        self.features['patterns'] = patterns
        return patterns
    
    def generate_volume_features(self) -> Dict:
        """
        Extract volume profile features.
        
        Returns:
            Dictionary of volume features
        """
        volume = {}
        vol_ma = self.df['Volume'].rolling(window=20).mean()
        
        for i in range(1, len(self.df)):
            v = self.df.loc[i, 'Volume']
            v_ma = vol_ma.iloc[i] if i < len(vol_ma) else v
            c = self.df.loc[i, 'Close']
            c_prev = self.df.loc[i-1, 'Close']
            
            volume[f'vol_surge_{i}'] = 1 if v > v_ma * 1.3 else 0
            volume[f'vol_decline_{i}'] = 1 if v < v_ma * 0.7 else 0
            volume[f'vol_momentum_{i}'] = (v - v_ma) / v_ma if v_ma > 0 else 0
            volume[f'price_vol_ratio_{i}'] = abs(c - c_prev) * v / (v_ma + 0.001)
        
        self.features['volume'] = volume
        return volume
    
    def generate_temporal_features(self) -> Dict:
        """
        Generate time-based features (requires timestamp data).
        
        Returns:
            Dictionary of temporal features
        """
        temporal = {}
        
        if 'Timestamp' not in self.df.columns:
            temporal['note'] = 'Timestamp data not available'
            self.features['temporal'] = temporal
            return temporal
        
        for i in range(1, len(self.df)):
            if i > 0:
                time_fraction = i / len(self.df)
                temporal[f'time_fraction_{i}'] = time_fraction
                temporal[f'session_position_{i}'] = time_fraction
        
        self.features['temporal'] = temporal
        return temporal
    
    def generate_all_features(self) -> Dict:
        """
        Generate all available features.
        
        Returns:
            Dictionary containing all feature groups
        """
        self.generate_combo_features()
        self.generate_candlestick_features()
        self.generate_volume_features()
        self.generate_temporal_features()
        
        return self.features
    
    def get_feature_importance(self, target: np.ndarray, method: str = 'correlation') -> List[Tuple]:
        """
        Rank features by importance.
        
        Args:
            target: Target variable (1D array)
            method: 'correlation' or 'mutual_info'
        
        Returns:
            List of (feature_name, importance_score) tuples
        """
        flat_features = {}
        
        for group_name, group_features in self.features.items():
            for feat_name, feat_value in group_features.items():
                if isinstance(feat_value, (int, float)):
                    flat_features[f'{group_name}_{feat_name}'] = feat_value
        
        feature_df = pd.DataFrame(flat_features, index=[0]).T
        feature_df.columns = ['value']
        
        if method == 'correlation' and len(target) == len(feature_df):
            correlations = []
            for fname, fvals in flat_features.items():
                if isinstance(fvals, (list, np.ndarray)):
                    if len(fvals) == len(target):
                        corr = np.corrcoef(fvals, target)[0, 1]
                        correlations.append((fname, abs(corr)))
            
            return sorted(correlations, key=lambda x: x[1], reverse=True)
        
        return list(flat_features.items())
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert all features to DataFrame format.
        
        Returns:
            DataFrame with all features as columns
        """
        flat_features = {}
        
        for group_name, group_features in self.features.items():
            for feat_name, feat_value in group_features.items():
                if isinstance(feat_value, (int, float)):
                    flat_features[f'{group_name}_{feat_name}'] = feat_value
        
        return pd.DataFrame(flat_features, index=[0])


def load_ohlcv_data(filepath: str) -> Optional[pd.DataFrame]:
    """
    Load OHLCV data from CSV file.
    
    Expected columns: Open, High, Low, Close, Volume
    """
    try:
        df = pd.read_csv(filepath)
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        if not all(col in df.columns for col in required_cols):
            print(f"Error: CSV must contain columns: {required_cols}")
            return None
        
        return df[required_cols]
    except Exception as e:
        print(f"Error loading CSV: {str(e)}")
        return None


def main():
    """CLI interface for feature generation."""
    
    parser = argparse.ArgumentParser(
        description='Generate OHLCV features from local or HuggingFace data'
    )
    parser.add_argument(
        '--file',
        type=str,
        help='Path to local OHLCV CSV file'
    )
    parser.add_argument(
        '--symbol',
        type=str,
        default='BTCUSDT',
        help='Trading pair symbol (for HuggingFace loading)'
    )
    parser.add_argument(
        '--timeframe',
        type=str,
        default='15m',
        choices=['15m', '1h'],
        help='Timeframe for data (15m or 1h)'
    )
    parser.add_argument(
        '--lookback',
        type=int,
        default=100,
        help='Number of recent candles to use'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output CSV file path for features'
    )
    parser.add_argument(
        '--source',
        type=str,
        default='huggingface',
        choices=['huggingface', 'csv', 'demo'],
        help='Data source: huggingface, csv file, or demo'
    )
    parser.add_argument(
        '--list-symbols',
        action='store_true',
        help='List all supported symbols'
    )
    
    args = parser.parse_args()
    
    if args.list_symbols:
        print("Supported symbols in HuggingFace dataset:")
        for i, sym in enumerate(HuggingFaceDataLoader.SUPPORTED_SYMBOLS, 1):
            print(f"  {i:2d}. {sym}")
        return
    
    df = None
    
    if args.source == 'csv' and args.file:
        print(f"Loading data from CSV: {args.file}")
        df = load_ohlcv_data(args.file)
        
    elif args.source == 'huggingface':
        if not HF_AVAILABLE:
            print("Error: huggingface_hub not installed")
            print("Install with: pip install huggingface_hub")
            return
        
        try:
            print(f"Loading {args.symbol} {args.timeframe} from HuggingFace...")
            df = HuggingFaceDataLoader.load_klines(args.symbol, args.timeframe)
            df = HuggingFaceDataLoader.standardize_columns(df)
        except Exception as e:
            print(f"Error loading from HuggingFace: {str(e)}")
            return
            
    elif args.source == 'demo':
        print("Generating sample data for demonstration...")
        np.random.seed(42)
        n = max(args.lookback, 100)
        df = pd.DataFrame({
            'Open': np.random.normal(100, 5, n),
            'High': np.random.normal(102, 5, n),
            'Low': np.random.normal(98, 5, n),
            'Close': np.random.normal(100, 5, n),
            'Volume': np.random.normal(1000, 200, n).astype(int)
        })
    
    if df is None or df.empty:
        print("Failed to load data. Please check your input.")
        return
    
    print(f"\nGenerating features for {args.symbol} {args.timeframe}...")
    print(f"Using {len(df)} candles with lookback={args.lookback}")
    
    generator = FeatureGenerator(df, lookback=args.lookback)
    features = generator.generate_all_features()
    
    print("\nFeature generation complete!")
    print(f"Total feature groups: {len(features)}")
    
    for group_name, group_features in features.items():
        if group_name != 'temporal' or 'note' not in group_features:
            print(f"  - {group_name}: {len(group_features)} features")
    
    feature_df = generator.to_dataframe()
    print(f"\nTotal features generated: {len(feature_df.columns)}")
    
    if args.output:
        feature_df.to_csv(args.output, index=False)
        print(f"Features saved to: {args.output}")
    else:
        print("\nSample features (first 10 columns):")
        print(feature_df.iloc[:, :10])


if __name__ == "__main__":
    main()
