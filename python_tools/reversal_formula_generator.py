#!/usr/bin/env python3

"""
Reversal Formula Generator - Reverse Engineer from Real Data

Method:
1. User manually marks actual reversal points in data
2. Extract OHLCV features from those reversal candles
3. Machine learning finds common pattern in reversals
4. Generate mathematical formula to predict reversals
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    import sympy as sp
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class ReversalDataLabeler:
    """Tools for marking and managing reversal points."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize labeler.
        
        Args:
            df: DataFrame with OHLCV data
        """
        self.df = df.copy()
        self.reversals = np.zeros(len(df), dtype=int)  # 0=not reversal, 1=reversal
        self.reversal_times = []
    
    def mark_reversal(self, index: int, reversal_type: str = 'bullish'):
        """
        Mark a candle as reversal point.
        
        Args:
            index: Candle index
            reversal_type: 'bullish' or 'bearish'
        """
        if 0 <= index < len(self.df):
            self.reversals[index] = 1
            self.reversal_times.append({
                'index': index,
                'type': reversal_type,
                'open': self.df.iloc[index]['Open'],
                'close': self.df.iloc[index]['Close'],
                'time': self.df.iloc[index].name if hasattr(self.df.iloc[index].name, '__str__') else str(index)
            })
    
    def mark_reversals_from_list(self, indices: List[int]):
        """
        Mark multiple reversals at once.
        
        Args:
            indices: List of candle indices that are reversals
        """
        for idx in indices:
            self.mark_reversal(idx)
    
    def load_reversals_from_json(self, json_file: str):
        """
        Load previously saved reversal marks.
        
        Args:
            json_file: Path to JSON file with reversal indices
        """
        with open(json_file, 'r') as f:
            data = json.load(f)
            indices = data.get('reversal_indices', [])
            self.mark_reversals_from_list(indices)
    
    def save_reversals_to_json(self, json_file: str):
        """
        Save marked reversals to JSON.
        
        Args:
            json_file: Path to save JSON
        """
        reversals = {
            'reversal_indices': np.where(self.reversals == 1)[0].tolist(),
            'reversal_count': int(np.sum(self.reversals)),
            'reversal_details': self.reversal_times
        }
        with open(json_file, 'w') as f:
            json.dump(reversals, f, indent=2, default=str)
    
    def get_reversal_candles(self) -> pd.DataFrame:
        """
        Get all candles marked as reversals.
        
        Returns:
            DataFrame of reversal candles
        """
        return self.df[self.reversals == 1]
    
    def get_non_reversal_candles(self) -> pd.DataFrame:
        """
        Get all non-reversal candles.
        
        Returns:
            DataFrame of non-reversal candles
        """
        return self.df[self.reversals == 0]


class ReversalFeatureExtractor:
    """Extract features from OHLCV candles."""
    
    @staticmethod
    def extract_features(candle_row: pd.Series) -> Dict[str, float]:
        """
        Extract all relevant features from single candle.
        
        Args:
            candle_row: Single row from OHLCV DataFrame
        
        Returns:
            Dictionary of features
        """
        o = candle_row['Open']
        h = candle_row['High']
        l = candle_row['Low']
        c = candle_row['Close']
        v = candle_row['Volume']
        
        features = {}
        
        # Body ratio
        range_hl = h - l
        if range_hl > 0:
            features['body_ratio'] = (c - o) / range_hl  # -1 to 1
            features['upper_wick_ratio'] = (h - max(o, c)) / range_hl
            features['lower_wick_ratio'] = (min(o, c) - l) / range_hl
            features['close_position'] = (c - l) / range_hl  # Where close is in range
            features['open_position'] = (o - l) / range_hl   # Where open is in range
        else:
            features['body_ratio'] = 0
            features['upper_wick_ratio'] = 0
            features['lower_wick_ratio'] = 0
            features['close_position'] = 0.5
            features['open_position'] = 0.5
        
        # Candle size
        features['body_size'] = abs(c - o) / o if o > 0 else 0
        features['total_range'] = range_hl / o if o > 0 else 0
        features['wick_total'] = (features['upper_wick_ratio'] + features['lower_wick_ratio'])
        
        # Momentum
        features['price_change_pct'] = (c - o) / o if o > 0 else 0
        
        # Volume
        features['volume'] = v
        features['log_volume'] = np.log(v) if v > 0 else 0
        
        # Pattern indicators
        is_bullish = c > o
        is_bearish = c < o
        is_doji = abs(c - o) < range_hl * 0.1
        has_upper_wick = features['upper_wick_ratio'] > 0.2
        has_lower_wick = features['lower_wick_ratio'] > 0.2
        
        features['is_bullish'] = 1 if is_bullish else 0
        features['is_bearish'] = 1 if is_bearish else 0
        features['is_doji'] = 1 if is_doji else 0
        features['has_long_upper_wick'] = 1 if has_upper_wick else 0
        features['has_long_lower_wick'] = 1 if has_lower_wick else 0
        features['hammer_pattern'] = 1 if (is_bullish and has_lower_wick and features['upper_wick_ratio'] < 0.1) else 0
        features['hanging_man_pattern'] = 1 if (is_bearish and has_lower_wick and features['upper_wick_ratio'] < 0.1) else 0
        features['shooting_star_pattern'] = 1 if (is_bearish and has_upper_wick and features['lower_wick_ratio'] < 0.1) else 0
        features['inverted_hammer_pattern'] = 1 if (is_bullish and has_upper_wick and features['lower_wick_ratio'] < 0.1) else 0
        
        return features
    
    @staticmethod
    def extract_batch_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from all candles.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with extracted features
        """
        features_list = []
        for idx, row in df.iterrows():
            features = ReversalFeatureExtractor.extract_features(row)
            features_list.append(features)
        
        return pd.DataFrame(features_list)


class ReversalFormulaDiscovery:
    """Discover formula from reversal features using ML."""
    
    def __init__(self, reversal_features: pd.DataFrame, non_reversal_features: pd.DataFrame):
        """
        Initialize with reversal and non-reversal features.
        
        Args:
            reversal_features: Features from actual reversal candles
            non_reversal_features: Features from normal candles
        """
        self.reversal_features = reversal_features.copy()
        self.non_reversal_features = non_reversal_features.copy()
        
        # Prepare training data
        reversal_features['label'] = 1
        non_reversal_features['label'] = 0
        
        self.training_data = pd.concat([reversal_features, non_reversal_features], ignore_index=True)
        self.feature_names = [col for col in self.training_data.columns if col != 'label']
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
    
    def train_model(self) -> Dict:
        """
        Train random forest to find pattern.
        
        Returns:
            Training metrics
        """
        X = self.training_data[self.feature_names].fillna(0)
        y = self.training_data['label']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42
        )
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Get feature importance
        self.feature_importance = dict(zip(
            self.feature_names,
            self.model.feature_importances_
        ))
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'feature_importance': self.feature_importance
        }
    
    def get_top_features(self, top_n: int = 5) -> List[Tuple]:
        """
        Get most important features.
        
        Args:
            top_n: Number of top features
        
        Returns:
            List of (feature_name, importance) sorted by importance
        """
        if self.feature_importance is None:
            return []
        
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_features[:top_n]
    
    def generate_formula_string(self, top_n: int = 3) -> str:
        """
        Generate human-readable formula from top features.
        
        Args:
            top_n: Number of features to use in formula
        
        Returns:
            Formula string
        """
        top_features = self.get_top_features(top_n)
        
        formula_parts = []
        for feature, importance in top_features:
            # Get threshold from reversal candles
            reversal_mean = self.reversal_features[feature].mean()
            reversal_std = self.reversal_features[feature].std()
            
            non_reversal_mean = self.non_reversal_features[feature].mean()
            
            # Calculate threshold
            threshold = (reversal_mean + non_reversal_mean) / 2
            
            if reversal_mean > non_reversal_mean:
                formula_parts.append(f"({feature} > {threshold:.4f})")
            else:
                formula_parts.append(f"({feature} < {threshold:.4f})")
        
        formula = " AND ".join(formula_parts)
        return formula
    
    def generate_pine_script(self, top_n: int = 3) -> str:
        """
        Generate Pine Script code from discovered formula.
        
        Args:
            top_n: Number of features to use
        
        Returns:
            Pine Script code
        """
        top_features = self.get_top_features(top_n)
        
        pine_code = """// Auto-generated Reversal Detection Formula
// Based on analysis of actual reversal points

@indicator("Reversal Formula", overlay=false)

// Extract features
bodyRatio = (close - open) / (high - low + 0.001)
upperWickRatio = (high - math.max(open, close)) / (high - low + 0.001)
lowerWickRatio = (math.min(open, close) - low) / (high - low + 0.001)
closePosition = (close - low) / (high - low + 0.001)
bodySize = math.abs(close - open) / open
totalRange = (high - low) / open

"""
        
        conditions = []
        for feature, importance in top_features:
            reversal_mean = self.reversal_features[feature].mean()
            non_reversal_mean = self.non_reversal_features[feature].mean()
            threshold = (reversal_mean + non_reversal_mean) / 2
            
            if feature == 'body_ratio':
                if reversal_mean > non_reversal_mean:
                    conditions.append(f"bodyRatio > {threshold:.4f}")
                else:
                    conditions.append(f"bodyRatio < {threshold:.4f}")
            
            elif feature == 'upper_wick_ratio':
                if reversal_mean > non_reversal_mean:
                    conditions.append(f"upperWickRatio > {threshold:.4f}")
                else:
                    conditions.append(f"upperWickRatio < {threshold:.4f}")
            
            # Add more feature mappings as needed
        
        condition_string = " and ".join(conditions) if conditions else "true"
        
        pine_code += f"""
// Reversal detection logic
reversalSignal = {condition_string}

plot(reversalSignal ? 1 : 0, title="Reversal Signal", color=reversalSignal ? color.green : color.red)
"""
        
        return pine_code


def create_example_reversal_file():
    """
    Create example JSON file showing how to mark reversals.
    """
    example = {
        "instructions": "List the indices (row numbers) of candles you identified as reversal points",
        "reversal_indices": [45, 89, 134, 178, 223, 267, 312, 356],
        "notes": "You can get indices by looking at candle numbers in TradingView or Excel"
    }
    
    with open('reversal_marks_example.json', 'w') as f:
        json.dump(example, f, indent=2)
    
    print("Created reversal_marks_example.json")


def main():
    """CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Reverse-engineer reversal formula from actual reversal points'
    )
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading pair')
    parser.add_argument('--lookback', type=int, default=500, help='Number of candles')
    parser.add_argument('--reversals', type=str, help='JSON file with marked reversal indices')
    parser.add_argument('--create-example', action='store_true', help='Create example reversal marks file')
    parser.add_argument('--output', type=str, help='Save formula to file')
    
    args = parser.parse_args()
    
    if args.create_example:
        create_example_reversal_file()
        return
    
    # Load data
    print(f"Loading {args.symbol} data...")
    try:
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq
        
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
    
    # Load reversal marks or use example
    if args.reversals:
        print(f"Loading reversal marks from {args.reversals}...")
        labeler = ReversalDataLabeler(df)
        labeler.load_reversals_from_json(args.reversals)
    else:
        print("No reversal marks provided. Use --reversals <file.json>")
        print("\nTo create your reversal marks:")
        print("1. Run: python reversal_formula_generator.py --create-example")
        print("2. Edit reversal_marks_example.json with your reversal indices")
        print("3. Run: python reversal_formula_generator.py --reversals reversal_marks_example.json")
        return
    
    reversal_count = np.sum(labeler.reversals)
    print(f"Found {reversal_count} reversal points\n")
    
    if reversal_count < 3:
        print("Error: Need at least 3 reversal points to discover formula")
        return
    
    # Extract features
    print("Extracting features from reversal candles...")
    reversal_candles = labeler.get_reversal_candles()
    non_reversal_candles = labeler.get_non_reversal_candles()
    
    reversal_features = ReversalFeatureExtractor.extract_batch_features(reversal_candles)
    non_reversal_features = ReversalFeatureExtractor.extract_batch_features(non_reversal_candles)
    
    print(f"Reversal candles: {len(reversal_features)}")
    print(f"Non-reversal candles: {len(non_reversal_features)}\n")
    
    # Discover formula
    print("Training model to discover formula...\n")
    discovery = ReversalFormulaDiscovery(reversal_features, non_reversal_features)
    metrics = discovery.train_model()
    
    print(f"Model Accuracy:")
    print(f"  Train: {metrics['train_accuracy']:.4f}")
    print(f"  Test:  {metrics['test_accuracy']:.4f}\n")
    
    # Show top features
    print("Top 5 Most Important Reversal Features:")
    print("=" * 60)
    for i, (feature, importance) in enumerate(discovery.get_top_features(5), 1):
        reversal_mean = reversal_features[feature].mean()
        non_reversal_mean = non_reversal_features[feature].mean()
        print(f"\n{i}. {feature} (importance: {importance:.4f})")
        print(f"   Reversal candles avg: {reversal_mean:.4f}")
        print(f"   Normal candles avg:   {non_reversal_mean:.4f}")
    
    # Generate formula
    print("\n" + "=" * 60)
    print("\nDiscovered Reversal Formula:")
    print("=" * 60)
    formula = discovery.generate_formula_string(top_n=3)
    print(f"\n{formula}\n")
    
    # Generate Pine Script
    print("\nGenerated Pine Script:")
    print("=" * 60)
    pine_code = discovery.generate_pine_script(top_n=3)
    print(pine_code)
    
    # Save if requested
    if args.output:
        with open(args.output, 'w') as f:
            f.write(pine_code)
        print(f"\nPine Script saved to: {args.output}")


if __name__ == "__main__":
    main()
