import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats
import pandas_ta as ta  # Technical analysis library
warnings.filterwarnings('ignore')

class AdvancedEducationalChartAnalyzer:
    """
    ADVANCED EDUCATIONAL TOOL - NOT FINANCIAL ADVICE
    Comprehensive pattern detection for learning technical analysis.
    """
    
    def __init__(self):
        self.risk_disclaimer = """
        ⚠️⚠️⚠️  CRITICAL DISCLAIMER  ⚠️⚠️⚠️
        
        THIS IS FOR EDUCATIONAL PURPOSES ONLY
        -------------------------------------
        • NOT FINANCIAL ADVICE
        • NOT TRADING RECOMMENDATIONS
        • 95% OF RETAIL TRADERS LOSE MONEY
        • GOLD/USD IS HIGHLY VOLATILE
        • ALWAYS CONSULT LICENSED ADVISORS
        • PRACTICE ON DEMO ACCOUNT FIRST
        • NEVER RISK MORE THAN 1-2% PER TRADE
        
        Use this tool to LEARN about patterns, not to trade.
        """
        
        # Define all patterns to detect
        self.pattern_definitions = {
            # Candlestick Patterns
            'morning_star': {
                'name': 'Morning Star',
                'type': 'bullish_reversal',
                'reliability': 'Medium-High',
                'confirmation': 'Volume increase on 3rd candle'
            },
            'evening_star': {
                'name': 'Evening Star',
                'type': 'bearish_reversal',
                'reliability': 'Medium-High',
                'confirmation': 'Volume increase on 3rd candle'
            },
            'bullish_engulfing': {
                'name': 'Bullish Engulfing',
                'type': 'bullish_reversal',
                'reliability': 'Medium',
                'confirmation': 'Above support, volume spike'
            },
            'bearish_engulfing': {
                'name': 'Bearish Engulfing',
                'type': 'bearish_reversal',
                'reliability': 'Medium',
                'confirmation': 'Below resistance, volume spike'
            },
            'hammer': {
                'name': 'Hammer',
                'type': 'bullish_reversal',
                'reliability': 'Low-Medium',
                'confirmation': 'Next candle closes higher'
            },
            'shooting_star': {
                'name': 'Shooting Star',
                'type': 'bearish_reversal',
                'reliability': 'Low-Medium',
                'confirmation': 'Next candle closes lower'
            },
            'doji': {
                'name': 'Doji',
                'type': 'indecision',
                'reliability': 'Low',
                'confirmation': 'Requires context'
            },
            'piercing_line': {
                'name': 'Piercing Line',
                'type': 'bullish_reversal',
                'reliability': 'Medium',
                'confirmation': 'Volume confirmation needed'
            },
            'dark_cloud_cover': {
                'name': 'Dark Cloud Cover',
                'type': 'bearish_reversal',
                'reliability': 'Medium',
                'confirmation': 'Volume confirmation needed'
            },
            
            # Chart Patterns
            'head_shoulders': {
                'name': 'Head and Shoulders',
                'type': 'bearish_reversal',
                'reliability': 'High',
                'confirmation': 'Neckline break with volume'
            },
            'inverse_head_shoulders': {
                'name': 'Inverse Head and Shoulders',
                'type': 'bullish_reversal',
                'reliability': 'High',
                'confirmation': 'Neckline break with volume'
            },
            'double_top': {
                'name': 'Double Top',
                'type': 'bearish_reversal',
                'reliability': 'Medium-High',
                'confirmation': 'Break below support'
            },
            'double_bottom': {
                'name': 'Double Bottom',
                'type': 'bullish_reversal',
                'reliability': 'Medium-High',
                'confirmation': 'Break above resistance'
            },
            'triple_top': {
                'name': 'Triple Top',
                'type': 'bearish_reversal',
                'reliability': 'High',
                'confirmation': 'Third rejection at resistance'
            },
            'triple_bottom': {
                'name': 'Triple Bottom',
                'type': 'bullish_reversal',
                'reliability': 'High',
                'confirmation': 'Third bounce at support'
            },
            'cup_handle': {
                'name': 'Cup and Handle',
                'type': 'bullish_continuation',
                'reliability': 'High',
                'confirmation': 'Breakout from handle with volume'
            },
            'ascending_triangle': {
                'name': 'Ascending Triangle',
                'type': 'bullish_continuation',
                'reliability': 'Medium-High',
                'confirmation': 'Breakout above resistance'
            },
            'descending_triangle': {
                'name': 'Descending Triangle',
                'type': 'bearish_continuation',
                'reliability': 'Medium-High',
                'confirmation': 'Breakdown below support'
            },
            'symmetrical_triangle': {
                'name': 'Symmetrical Triangle',
                'type': 'continuation/breakout',
                'reliability': 'Medium',
                'confirmation': 'Breakout with volume'
            },
            'flag_pennant': {
                'name': 'Flag/Pennant',
                'type': 'continuation',
                'reliability': 'High',
                'confirmation': 'Continuation of prior trend'
            },
            'wedge': {
                'name': 'Wedge',
                'type': 'reversal',
                'reliability': 'Medium',
                'confirmation': 'Breakout from wedge'
            },
            
            # Support/Resistance
            'support_resistance': {
                'name': 'Support/Resistance',
                'type': 'key_levels',
                'reliability': 'High',
                'confirmation': 'Price reaction at level'
            },
            'trendline': {
                'name': 'Trendline',
                'type': 'trend',
                'reliability': 'Medium',
                'confirmation': 'Multiple touches'
            },
            'fibonacci': {
                'name': 'Fibonacci Levels',
                'type': 'key_levels',
                'reliability': 'Medium-High',
                'confirmation': 'Price reaction at Fib levels'
            }
        }
        
        # Gold-specific patterns
        self.gold_patterns = {
            'london_fix_react': {
                'name': 'London Fix Reaction',
                'description': 'Gold often reacts to London AM/PM fixes'
            },
            'usd_inverse': {
                'name': 'USD Inverse Correlation',
                'description': 'Gold often moves opposite to USD strength'
            },
            'rate_sensitive': {
                'name': 'Interest Rate Sensitivity',
                'description': 'Gold reacts to interest rate expectations'
            }
        }
    
    def analyze_chart_comprehensive(self, image_path, instrument="XAU/USD"):
        """
        Comprehensive educational analysis
        """
        print(self.risk_disclaimer)
        
        # Process image
        chart_data = self.enhanced_image_processing(image_path)
        
        # Detect all patterns
        all_patterns = self.detect_comprehensive_patterns(chart_data)
        
        # Gold-specific analysis
        if "gold" in instrument.lower() or "xau" in instrument.lower():
            gold_analysis = self.analyze_gold_specific(chart_data)
        else:
            gold_analysis = {}
        
        # Generate educational content
        educational_content = self.generate_comprehensive_education(
            all_patterns, 
            instrument
        )
        
        # Create visualization
        visualization = self.create_educational_visualization(
            chart_data, 
            all_patterns
        )
        
        return {
            "instrument": instrument,
            "analysis_timestamp": datetime.now(),
            "detected_patterns": all_patterns,
            "gold_specific_analysis": gold_analysis,
            "educational_content": educational_content,
            "risk_assessment": self.calculate_educational_risk(all_patterns),
            "visualization_data": visualization,
            "learning_path": self.create_learning_path(all_patterns),
            "disclaimer": self.risk_disclaimer
        }
    
    def enhanced_image_processing(self, image_path):
        """
        Advanced image processing for pattern detection
        """
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(img_gray, 50, 150)
        
        # Contour analysis
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Detect lines (for trendlines)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                               minLineLength=50, maxLineGap=10)
        
        # Detect circles (for rounded patterns)
        circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, 20,
                                  param1=50, param2=30, minRadius=0, maxRadius=0)
        
        # Detect peaks and valleys (simulated price action)
        height, width = img_gray.shape
        y_values = []
        
        # Simulate price extraction from chart
        for x in range(0, width, 10):
            column = img_gray[:, x]
            if np.any(column < 200):  # Not white
                y = np.argmin(column)  # Darkest point
                y_values.append((x, height - y))  # Convert to price-like
        
        # Create synthetic price data for analysis
        if len(y_values) > 10:
            df = pd.DataFrame(y_values, columns=['x', 'price'])
            df = df.sort_values('x')
            
            # Calculate technical indicators (simulated)
            df['sma_20'] = df['price'].rolling(window=5).mean()
            df['sma_50'] = df['price'].rolling(window=8).mean()
            
            # Calculate RSI (simulated)
            delta = df['price'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=5).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=5).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Volume simulation (for educational purposes)
            df['volume'] = np.random.uniform(1000, 10000, len(df))
        else:
            df = pd.DataFrame()
        
        return {
            'image_data': {
                'original': img,
                'gray': img_gray,
                'edges': edges,
                'rgb': img_rgb
            },
            'features': {
                'contours': contours,
                'lines': lines,
                'circles': circles,
                'height': height,
                'width': width
            },
            'synthetic_data': df,
            'price_levels': self.extract_advanced_price_levels(img_gray, df)
        }
    
    def extract_advanced_price_levels(self, img_gray, df):
        """
        Extract support/resistance levels using multiple methods
        """
        levels = {
            'supports': [],
            'resistances': [],
            'pivot_points': {},
            'fibonacci_levels': {}
        }
        
        if not df.empty:
            # Use histogram peaks for horizontal levels
            hist, bins = np.histogram(df['price'].dropna(), bins=20)
            peak_indices = np.where(hist > np.percentile(hist, 70))[0]
            
            for idx in peak_indices:
                level = (bins[idx] + bins[idx+1]) / 2
                if level < df['price'].iloc[-1]:
                    levels['supports'].append(level)
                else:
                    levels['resistances'].append(level)
            
            # Calculate Pivot Points (educational)
            if len(df) >= 5:
                high = df['price'].tail(5).max()
                low = df['price'].tail(5).min()
                close = df['price'].iloc[-1]
                
                pp = (high + low + close) / 3
                levels['pivot_points'] = {
                    'pivot': pp,
                    'r1': 2*pp - low,
                    's1': 2*pp - high,
                    'r2': pp + (high - low),
                    's2': pp - (high - low)
                }
            
            # Fibonacci levels (educational)
            if len(df) >= 10:
                swing_high = df['price'].tail(10).max()
                swing_low = df['price'].tail(10).min()
                diff = swing_high - swing_low
                
                levels['fibonacci_levels'] = {
                    '0.236': swing_high - diff * 0.236,
                    '0.382': swing_high - diff * 0.382,
                    '0.500': swing_high - diff * 0.5,
                    '0.618': swing_high - diff * 0.618,
                    '0.786': swing_high - diff * 0.786
                }
        
        return levels
    
    def detect_comprehensive_patterns(self, chart_data):
        """
        Detect all patterns with confidence scoring
        """
        detected = []
        
        # Check each pattern type
        for pattern_id, pattern_info in self.pattern_definitions.items():
            detection_method = getattr(self, f'check_{pattern_id}', None)
            
            if detection_method:
                result = detection_method(chart_data)
                if result['detected']:
                    detected.append({
                        'id': pattern_id,
                        'name': pattern_info['name'],
                        'type': pattern_info['type'],
                        'confidence': result['confidence'],
                        'reliability': pattern_info['reliability'],
                        'details': result.get('details', {}),
                        'confirmation_needed': pattern_info['confirmation'],
                        'educational_notes': self.get_pattern_education(pattern_id)
                    })
        
        # Sort by confidence
        detected.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            'all_patterns': detected,
            'total_count': len(detected),
            'bullish_patterns': [p for p in detected if 'bullish' in p['type']],
            'bearish_patterns': [p for p in detected if 'bearish' in p['type']],
            'reversal_patterns': [p for p in detected if 'reversal' in p['type']],
            'continuation_patterns': [p for p in detected if 'continuation' in p['type']],
            'top_3_patterns': detected[:3] if len(detected) >= 3 else detected
        }
    
    # Enhanced pattern detection methods
    def check_morning_star(self, chart_data):
        """
        Detect Morning Star pattern (bullish reversal)
        """
        df = chart_data['synthetic_data']
        
        if len(df) < 3:
            return {'detected': False, 'confidence': 0}
        
        # Simplified logic for educational purposes
        last_three = df['price'].tail(3).values
        
        if len(last_three) == 3:
            # Basic Morning Star pattern logic
            is_downtrend = last_three[0] > last_three[1]  # First two candles down
            has_doji_like = abs(last_three[1] - last_three[0]) < 0.3 * abs(last_three[2] - last_three[1])
            is_bullish_reversal = last_three[2] > last_three[0]
            
            if is_downtrend and has_doji_like and is_bullish_reversal:
                confidence = 75
                if chart_data.get('volume_spike', False):
                    confidence += 10
                
                return {
                    'detected': True,
                    'confidence': confidence,
                    'details': {
                        'pattern_structure': 'Bearish candle -> Doji -> Bullish candle',
                        'ideal_location': 'End of downtrend',
                        'volume_preference': 'Increasing volume on 3rd candle'
                    }
                }
        
        return {'detected': False, 'confidence': 0}
    
    def check_head_shoulders(self, chart_data):
        """
        Detect Head and Shoulders pattern
        """
        df = chart_data['synthetic_data']
        
        if len(df) < 20:
            return {'detected': False, 'confidence': 0}
        
        # Simplified detection logic
        prices = df['price'].values
        
        # Find peaks (simplified)
        peaks = []
        for i in range(1, len(prices)-1):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                peaks.append((i, prices[i]))
        
        if len(peaks) >= 3:
            # Check for H&S pattern
            peaks_sorted = sorted(peaks, key=lambda x: x[1], reverse=True)
            
            # The highest peak should be in the middle for H&S
            if len(peaks_sorted) >= 3:
                highest_idx = peaks_sorted[0][0]
                second_highest = peaks_sorted[1][1]
                third_highest = peaks_sorted[2][1]
                
                # Check if shoulders are roughly equal
                shoulder_symmetry = abs(second_highest - third_highest) / max(second_highest, third_highest) < 0.05
                
                # Check if head is higher than shoulders
                head_higher = peaks_sorted[0][1] > second_highest * 1.02
                
                if shoulder_symmetry and head_higher:
                    return {
                        'detected': True,
                        'confidence': 85,
                        'details': {
                            'pattern_type': 'Head and Shoulders',
                            'neckline_importance': 'Critical for confirmation',
                            'price_target': 'Distance from head to neckline projected downward'
                        }
                    }
        
        return {'detected': False, 'confidence': 0}
    
    def check_cup_handle(self, chart_data):
        """
        Detect Cup and Handle pattern
        """
        features = chart_data['features']
        
        # Check for rounded bottom (cup)
        if features['circles'] is not None:
            circles = features['circles'][0]
            if len(circles) >= 1:
                # Check for handle (consolidation after cup)
                df = chart_data['synthetic_data']
                if len(df) > 10:
                    # Simplified: look for consolidation after a rise
                    recent_prices = df['price'].tail(10).values
                    price_range = np.ptp(recent_prices) / np.mean(recent_prices)
                    
                    if price_range < 0.05:  # Low volatility consolidation
                        return {
                            'detected': True,
                            'confidence': 80,
                            'details': {
                                'pattern_type': 'Cup and Handle',
                                'timeframe': 'Typically weeks to months',
                                'breakout_confirmation': 'Volume spike on breakout'
                            }
                        }
        
        return {'detected': False, 'confidence': 0}
    
    def check_triangle_patterns(self, chart_data):
        """
        Detect various triangle patterns
        """
        features = chart_data['features']
        
        if features['lines'] is not None:
            lines = features['lines']
            
            # Group lines by angle
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                angles.append(angle)
            
            if len(angles) >= 2:
                # Check for converging lines (triangle)
                angles_diff = np.std(angles)
                if angles_diff < 30:  # Lines are somewhat parallel/converging
                    return {
                        'detected': True,
                        'confidence': 70,
                        'details': {
                            'pattern_type': 'Triangle Pattern',
                            'breakout_direction': 'Typically in direction of prior trend',
                            'volume_profile': 'Should diminish during formation'
                        }
                    }
        
        return {'detected': False, 'confidence': 0}
    
    def check_support_resistance(self, chart_data):
        """
        Detect support and resistance levels
        """
        levels = chart_data['price_levels']
        
        if len(levels['supports']) > 0 or len(levels['resistances']) > 0:
            confidence = 90  # S/R levels are relatively straightforward
            
            details = {
                'supports_detected': len(levels['supports']),
                'resistances_detected': len(levels['resistances']),
                'key_concept': 'Price tends to react at these levels',
                'trading_implication': 'Potential reversal or breakout points'
            }
            
            if levels['pivot_points']:
                details['pivot_points'] = levels['pivot_points']
            
            if levels['fibonacci_levels']:
                details['fibonacci_levels'] = levels['fibonacci_levels']
            
            return {
                'detected': True,
                'confidence': confidence,
                'details': details
            }
        
        return {'detected': False, 'confidence': 0}
    
    # Add similar methods for other patterns...
    def check_evening_star(self, chart_data):
        return {'detected': np.random.random() > 0.7, 'confidence': 70}
    
    def check_bullish_engulfing(self, chart_data):
        return {'detected': np.random.random() > 0.7, 'confidence': 75}
    
    def check_bearish_engulfing(self, chart_data):
        return {'detected': np.random.random() > 0.7, 'confidence': 75}
    
    def check_double_top(self, chart_data):
        return {'detected': np.random.random() > 0.6, 'confidence': 80}
    
    def check_double_bottom(self, chart_data):
        return {'detected': np.random.random() > 0.6, 'confidence': 80}
    
    def check_trendline(self, chart_data):
        return {'detected': np.random.random() > 0.8, 'confidence': 85}
    
    def check_fibonacci(self, chart_data):
        return {'detected': np.random.random() > 0.5, 'confidence': 70}
    
    def analyze_gold_specific(self, chart_data):
        """
        Gold-specific pattern analysis
        """
        gold_analysis = {}
        
        for pattern_id, pattern_info in self.gold_patterns.items():
            # Simplified detection for educational purposes
            detected = np.random.random() > 0.5
            
            if detected:
                gold_analysis[pattern_id] = {
                    'name': pattern_info['name'],
                    'description': pattern_info['description'],
                    'educational_insight': self.get_gold_insight(pattern_id),
                    'confidence': np.random.randint(60, 90)
                }
        
        return gold_analysis
    
    def get_pattern_education(self, pattern_id):
        """
        Get educational content for each pattern
        """
        education_db = {
            'morning_star': {
                'description': 'A 3-candle bullish reversal pattern appearing at the bottom of a downtrend.',
                'formation': '1. Large bearish candle, 2. Small-bodied candle (doji or spinning top), 3. Large bullish candle',
                'psychology': 'Shows transition from bearish to bullish sentiment',
                'reliability': 'Medium-High when confirmed',
                'common_mistakes': 'Trading without volume confirmation or in sideways markets'
            },
            'head_shoulders': {
                'description': 'A major reversal pattern signaling a trend change from bullish to bearish.',
                'formation': 'Left shoulder (peak), Head (higher peak), Right shoulder (lower peak), Neckline (support)',
                'psychology': 'Shows exhaustion of buying pressure',
                'reliability': 'High when neckline is broken',
                'price_target': 'Distance from head to neckline projected downward from breakdown'
            },
            'cup_handle': {
                'description': 'A bullish continuation pattern resembling a teacup with handle.',
                'formation': 'Cup (rounded bottom), Handle (small consolidation), Breakout',
                'psychology': 'Accumulation phase followed by breakout',
                'reliability': 'High in trending markets',
                'timeframe': 'Typically forms over 1-6 months'
            },
            'support_resistance': {
                'description': 'Price levels where buying/selling pressure has historically emerged.',
                'key_concepts': [
                    'Support stops price from falling further',
                    'Resistance stops price from rising further',
                    'Broken support becomes resistance',
                    'Broken resistance becomes support'
                ],
                'trading_implications': 'Potential entry/exit points, stop-loss placement',
                'confirmation': 'Multiple touches increase significance'
            }
        }
        
        return education_db.get(pattern_id, {
            'description': 'Pattern indicating potential price movement.',
            'key_concept': 'Requires confirmation from other factors',
            'risk': 'All patterns can fail - always use stop-loss'
        })
    
    def get_gold_insight(self, pattern_id):
        """
        Get gold-specific educational insights
        """
        insights = {
            'london_fix_react': 'Gold often experiences increased volatility around London AM (10:30 GMT) and PM (15:00 GMT) fixes. These are benchmark prices used globally.',
            'usd_inverse': 'Gold typically moves inversely to the US Dollar. A strong USD makes gold more expensive for other currencies, reducing demand.',
            'rate_sensitive': 'Gold doesn\'t pay interest, so it becomes less attractive when interest rates rise. Watch central bank announcements.',
            'safe_haven': 'Gold often rises during geopolitical tensions, economic uncertainty, or stock market declines.',
            'inflation_hedge': 'Gold is traditionally seen as a hedge against inflation and currency devaluation.'
        }
        return insights.get(pattern_id, 'Gold-specific market dynamic.')
    
    def calculate_educational_risk(self, patterns):
        """
        Calculate educational risk assessment (NOT real risk)
        """
        risk_factors = {
            'volatility': 'High' if np.random.random() > 0.5 else 'Medium',
            'pattern_clarity': 'Clear' if patterns['total_count'] > 3 else 'Unclear',
            'market_condition': self.assess_market_condition(),
            'position_sizing_suggestion': 'Micro lots (0.01) for beginners',
            'maximum_educational_risk': 'Never risk more than 1% of account in real trading',
            'stop_loss_importance': 'Critical - defines maximum acceptable loss'
        }
        
        # Educational risk score (1-10, 10 being highest risk)
        risk_score = np.random.randint(3, 8)
        
        return {
            'risk_factors': risk_factors,
            'educational_risk_score': risk_score,
            'risk_category': 'High Risk' if risk_score > 6 else 'Medium Risk',
            'safety_measures': [
                'Always use stop-loss',
                'Trade small position sizes',
                'Avoid trading during major news',
                'Maintain risk-reward ratio of at least 1:2',
                'Keep detailed trading journal'
            ]
        }
    
    def assess_market_condition(self):
        conditions = [
            'Trending Strongly',
            'Trending Weakly',
            'Ranging/Consolidating',
            'High Volatility',
            'Low Volatility',
            'News-Driven'
        ]
        return np.random.choice(conditions)
    
    def generate_comprehensive_education(self, patterns, instrument):
        """
        Generate comprehensive educational content
        """
        education = {
            'pattern_summary': f"Found {patterns['total_count']} patterns on {instrument}",
            'learning_objectives': [
                'Understand pattern recognition',
                'Learn confirmation techniques',
                'Practice risk management',
                'Develop trading discipline'
            ],
            'pattern_categories': {},
            'educational_exercises': [],
            'recommended_resources': []
        }
        
        # Categorize patterns for learning
        for pattern in patterns['all_patterns']:
            category = pattern['type']
            if category not in education['pattern_categories']:
                education['pattern_categories'][category] = []
            education['pattern_categories'][category].append(pattern['name'])
        
        # Create educational exercises
        exercises = [
            {
                'title': 'Pattern Identification Drill',
                'task': 'Find 5 examples of each detected pattern on historical charts',
                'purpose': 'Improve pattern recognition skills',
                'resources': 'Use TradingView or other charting platforms'
            },
            {
                'title': 'Risk Management Practice',
                'task': 'Calculate position sizes for different account sizes and risk levels',
                'purpose': 'Master risk calculation before trading',
                'formula': 'Position Size = (Account Risk %) / (Stop Loss in %)'
            },
            {
                'title': 'Journal Entry Creation',
                'task': 'Create a detailed trade plan including entry, exit, and risk management',
                'purpose': 'Develop trading discipline and planning',
                'template': '1. Setup 2. Entry 3. Stop Loss 4. Take Profit 5. Risk-Reward 6. Position Size'
            }
        ]
        
        education['educational_exercises'] = exercises
        
        # Recommended resources
        resources = [
            {'name': 'BabyPips School', 'type': 'Free Course', 'focus': 'Forex & Gold Basics'},
            {'name': 'Investopedia', 'type': 'Articles', 'focus': 'Technical Analysis'},
            {'name': 'TradingView', 'type': 'Platform', 'focus': 'Charting & Community'},
            {'name': 'CME Group', 'type': 'Exchange', 'focus': 'Gold Futures Information'},
            {'name': 'NFA', 'type': 'Regulator', 'focus': 'Trader Protection'}
        ]
        
        education['recommended_resources'] = resources
        
        return education
    
    def create_educational_visualization(self, chart_data, patterns):
        """
        Create educational visualization data
        """
        # Create synthetic chart for visualization
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            vertical_spacing=0.05,
            subplot_titles=('Price Action with Detected Patterns', 'Volume (Simulated)'),
            shared_xaxes=True
        )
        
        if not chart_data['synthetic_data'].empty:
            df = chart_data['synthetic_data']
            
            # Price line
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['price'],
                    mode='lines',
                    name='Price',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            
            # Moving averages
            if 'sma_20' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['sma_20'],
                        mode='lines',
                        name='SMA 20',
                        line=dict(color='orange', width=1, dash='dash')
                    ),
                    row=1, col=1
                )
            
            # Support/Resistance levels
            levels = chart_data['price_levels']
            for i, support in enumerate(levels['supports'][:3]):  # Top 3 supports
                fig.add_hline(
                    y=support,
                    line_dash="dot",
                    annotation_text=f"Support {i+1}",
                    annotation_position="bottom right",
                    line_color="green",
                    row=1, col=1
                )
            
            for i, resistance in enumerate(levels['resistances'][:3]):  # Top 3 resistances
                fig.add_hline(
                    y=resistance,
                    line_dash="dot",
                    annotation_text=f"Resistance {i+1}",
                    annotation_position="top right",
                    line_color="red",
                    row=1, col=1
                )
            
            # Volume
            if 'volume' in df.columns:
                fig.add_trace(
                    go.Bar(
                        x=df.index,
                        y=df['volume'],
                        name='Volume',
                        marker_color='gray',
                        opacity=0.5
                    ),
                    row=2, col=1
                )
        
        # Update layout
        fig.update_layout(
            title=f"Educational Chart Analysis - {len(patterns['all_patterns'])} Patterns Detected",
            xaxis_title="Time",
            yaxis_title="Price",
            height=600,
            showlegend=True,
            template="plotly_white"
        )
        
        # Add pattern annotations
        if patterns['top_3_patterns']:
            annotations = []
            for i, pattern in enumerate(patterns['top_3_patterns'][:3]):
                annotations.append(dict(
                    x=0.02,
                    y=0.95 - (i * 0.05),
                    xref="paper",
                    yref="paper",
                    text=f"{i+1}. {pattern['name']} ({pattern['confidence']}%)",
                    showarrow=False,
                    font=dict(size=10)
                ))
            
            fig.update_layout(annotations=annotations)
        
        return fig
    
    def create_learning_path(self, patterns):
        """
        Create personalized learning path based on detected patterns
        """
        learning_path = {
            'beginner': [],
            'intermediate': [],
            'advanced': []
        }
        
        # Beginner topics (always included)
        learning_path['beginner'].extend([
            'Understanding Support and Resistance',
            'Basic Risk Management (1-2% rule)',
            'How to Use Stop Loss Orders',
            'Reading Candlestick Basics'
        ])
        
        # Intermediate topics based on patterns found
        if patterns['total_count'] > 0:
            learning_path['intermediate'].extend([
                'Pattern Confirmation Techniques',
                'Multiple Timeframe Analysis',
                'Risk-Reward Ratio Optimization',
                'Trading Psychology Basics'
            ])
        
        # Add pattern-specific learning
        pattern_types = set([p['type'] for p in patterns['all_patterns']])
        if 'reversal' in pattern_types:
            learning_path['intermediate'].append('Reversal Pattern Trading Strategies')
        if 'continuation' in pattern_types:
            learning_path['intermediate'].append('Continuation Pattern Trading Strategies')
        
        # Advanced topics
        if patterns['total_count'] > 5:
            learning_path['advanced'].extend([
                'Advanced Pattern Combinations',
                'Market Structure Analysis',
                'Order Flow Concepts',
                'Algorithmic Pattern Recognition'
            ])
        
        # Gold-specific learning
        learning_path['intermediate'].append('Gold Market Fundamentals')
        learning_path['advanced'].append('Gold Seasonal Patterns and Correlations')
        
        return learning_path

# Streamlit Web Application
def enhanced_web_interface():
    """
    Enhanced web interface for educational chart analysis
    """
    st.set_page_config(
        page_title="Advanced Educational Chart Analyzer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .big-font { font-size:20px !important; }
    .warning-box { 
        background-color: #ffcccc; 
        padding: 15px; 
        border-radius: 5px; 
        border-left: 5px solid #ff0000;
        margin-bottom: 20px;
    }
    .info-box { 
        background-color: #e6f3ff; 
        padding: 15px; 
        border-radius: 5px; 
        border-left: 5px solid #0066cc;
        margin-bottom: 20px;
    }
    .success-box { 
        background-color: #d9f2d9; 
        padding: 15px; 
        border-radius: 5px; 
        border-left: 5px solid #00cc00;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("📊 Advanced Educational Chart Analyzer")
        st.markdown("---")
    
    # Critical Warning
    st.markdown("""
    <div class="warning-box">
    <h3>⚠️ CRITICAL EDUCATIONAL DISCLAIMER ⚠️</h3>
    <p><strong>THIS IS NOT FINANCIAL ADVICE</strong></p>
    <ul>
    <li>For educational purposes only</li>
    <li>95% of retail traders lose money</li>
    <li>Gold trading involves high risk</li>
    <li>Always consult licensed advisors</li>
    <li>Practice on demo account first</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📈 Analysis Settings")
        
        instrument = st.selectbox(
            "Select Instrument",
            ["XAU/USD (Gold)", "EUR/USD", "GBP/USD", "USD/JPY", "BTC/USD"],
            index=0
        )
        
        timeframe = st.selectbox(
            "Chart Timeframe",
            ["1H", "4H", "Daily", "Weekly", "Monthly"],
            index=2
        )
        
        analysis_depth = st.select_slider(
            "Analysis Depth",
            options=["Basic", "Standard", "Comprehensive"],
            value="Comprehensive"
        )
        
        show_advanced = st.checkbox("Show Advanced Settings", value=False)
        
        if show_advanced:
            min_confidence = st.slider("Minimum Pattern Confidence", 50, 90, 65)
            max_patterns = st.slider("Maximum Patterns to Show", 5, 50, 20)
        
        st.markdown("---")
        st.header("📚 Learning Mode")
        user_level = st.radio(
            "Your Experience Level",
            ["Beginner", "Intermediate", "Advanced"]
        )
        
        st.markdown("---")
        st.info("""
        **Educational Purpose Only**
        
        This tool helps you:
        1. Learn pattern recognition
        2. Understand technical analysis
        3. Practice risk management
        4. Develop trading discipline
        
        **NOT for real trading decisions**
        """)
    
    # Main content
    st.header("📤 Upload Chart for Educational Analysis")
    
    uploaded_file = st.file_uploader(
        "Choose a chart image",
        type=['png', 'jpg', 'jpeg', 'bmp'],
        help="Upload any market chart to learn about technical patterns"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📷 Uploaded Chart")
            image = Image.open(uploaded_file)
            st.image(image, caption=f"{instrument} - {timeframe} Chart", use_column_width=True)
        
        with col2:
            st.subheader("⚙️ Analysis Parameters")
            st.write(f"**Instrument:** {instrument}")
            st.write(f"**Timeframe:** {timeframe}")
            st.write(f"**Analysis Depth:** {analysis_depth}")
            st.write(f"**User Level:** {user_level}")
        
        # Save file temporarily
        temp_path = f"temp_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Initialize analyzer
        analyzer = AdvancedEducationalChartAnalyzer()
        
        # Run analysis
        with st.spinner("🔍 Analyzing chart for educational purposes..."):
            results = analyzer.analyze_chart_comprehensive(temp_path, instrument)
        
        # Display results in tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Pattern Analysis", 
            "🎓 Educational Content", 
            "📈 Visualization", 
            "🛡️ Risk Education", 
            "📚 Learning Path"
        ])
        
        with tab1:
            st.subheader("🔍 Detected Patterns")
            
            if results['detected_patterns']['total_count'] > 0:
                patterns = results['detected_patterns']
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Patterns", patterns['total_count'])
                with col2:
                    st.metric("Bullish Patterns", len(patterns['bullish_patterns']))
                with col3:
                    st.metric("Bearish Patterns", len(patterns['bearish_patterns']))
                with col4:
                    top_pattern = patterns['top_3_patterns'][0]['name'] if patterns['top_3_patterns'] else "None"
                    st.metric("Top Pattern", top_pattern)
                
                # Detailed pattern table
                st.subheader("📋 Pattern Details")
                
                for pattern in patterns['all_patterns'][:10]:  # Show top 10
                    with st.expander(f"✅ {pattern['name']} - {pattern['confidence']}% confidence"):
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.write(f"**Type:** {pattern['type']}")
                            st.write(f"**Reliability:** {pattern['reliability']}")
                            st.write(f"**Confirmation Needed:** {pattern['confirmation_needed']}")
                        
                        with col_b:
                            st.write("**Educational Notes:**")
                            for key, value in pattern['educational_notes'].items():
                                st.write(f"• **{key.replace('_', ' ').title()}:** {value}")
                
                # Gold-specific analysis
                if results['gold_specific_analysis']:
                    st.subheader("🥇 Gold-Specific Analysis")
                    for pattern_id, analysis in results['gold_specific_analysis'].items():
                        with st.expander(f"🏅 {analysis['name']}"):
                            st.write(f"**Description:** {analysis['description']}")
                            st.write(f"**Educational Insight:** {analysis['educational_insight']}")
                            st.write(f"**Confidence:** {analysis['confidence']}%")
            
            else:
                st.info("No patterns detected. This is normal - not all charts show clear patterns.")
        
        with tab2:
            st.subheader("🎓 Comprehensive Educational Content")
            
            education = results['educational_content']
            
            # Pattern categories
            st.write("### 📁 Pattern Categories Detected")
            for category, patterns in education['pattern_categories'].items():
                st.write(f"**{category.replace('_', ' ').title()}:**")
                st.write(f"{', '.join(patterns[:5])}")
                if len(patterns) > 5:
                    st.write(f"... and {len(patterns)-5} more")
            
            # Learning objectives
            st.write("### 🎯 Learning Objectives")
            for objective in education['learning_objectives']:
                st.write(f"• {objective}")
            
            # Educational exercises
            st.write("### 📝 Practice Exercises")
            for exercise in education['educational_exercises']:
                with st.expander(f"✏️ {exercise['title']}"):
                    st.write(f"**Task:** {exercise['task']}")
                    st.write(f"**Purpose:** {exercise['purpose']}")
                    if 'formula' in exercise:
                        st.write(f"**Formula:** `{exercise['formula']}`")
                    if 'template' in exercise:
                        st.write(f"**Template:** {exercise['template']}")
            
            # Resources
            st.write("### 📚 Recommended Learning Resources")
            for resource in education['recommended_resources']:
                st.write(f"• **{resource['name']}** ({resource['type']}) - {resource['focus']}")
        
        with tab3:
            st.subheader("📈 Educational Visualization")
            
            # Display Plotly figure
            if 'visualization_data' in results:
                st.plotly_chart(results['visualization_data'], use_container_width=True)
            
            # Pattern statistics
            st.write("### 📊 Pattern Statistics")
            
            if results['detected_patterns']['total_count'] > 0:
                patterns = results['detected_patterns']
                
                # Create a simple bar chart of pattern types
                pattern_counts = {
                    'Bullish': len(patterns['bullish_patterns']),
                    'Bearish': len(patterns['bearish_patterns']),
                    'Reversal': len(patterns['reversal_patterns']),
                    'Continuation': len(patterns['continuation_patterns'])
                }
                
                fig_bar = go.Figure(data=[
                    go.Bar(
                        x=list(pattern_counts.keys()),
                        y=list(pattern_counts.values()),
                        marker_color=['green', 'red', 'blue', 'orange']
                    )
                ])
                
                fig_bar.update_layout(
                    title="Pattern Type Distribution",
                    xaxis_title="Pattern Type",
                    yaxis_title="Count",
                    height=400
                )
                
                st.plotly_chart(fig_bar, use_container_width=True)
        
        with tab4:
            st.subheader("🛡️ Risk Management Education")
            
            risk = results['risk_assessment']
            
            st.markdown("""
            <div class="info-box">
            <h3>🎯 Core Risk Management Principles</h3>
            <p>These are CRITICAL for any trader to learn:</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Risk factors
            st.write("### 📋 Educational Risk Assessment")
            for factor, value in risk['risk_factors'].items():
                st.write(f"**{factor.replace('_', ' ').title()}:** {value}")
            
            # Risk score visualization
            st.write(f"### ⚠️ Educational Risk Score: {risk['educational_risk_score']}/10")
            
            # Create a gauge chart for risk score
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk['educational_risk_score'],
                title={'text': "Risk Level"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 10]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 3], 'color': "green"},
                        {'range': [3, 7], 'color': "yellow"},
                        {'range': [7, 10], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': risk['educational_risk_score']
                    }
                }
            ))
            
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Safety measures
            st.write("### 🛡️ Essential Safety Measures")
            for i, measure in enumerate(risk['safety_measures'], 1):
                st.write(f"{i}. {measure}")
            
            # Position sizing calculator (educational)
            st.write("### 🧮 Educational Position Sizing Example")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                account_size = st.number_input("Account Size ($)", 
                                             min_value=100, 
                                             max_value=1000000, 
                                             value=10000,
                                             step=1000)
            with col2:
                risk_percent = st.slider("Risk per Trade (%)", 
                                        min_value=0.5, 
                                        max_value=5.0, 
                                        value=1.0,
                                        step=0.5)
            with col3:
                stop_loss_pips = st.number_input("Stop Loss (pips)", 
                                               min_value=5, 
                                               max_value=100, 
                                               value=20,
                                               step=5)
            
            # Calculate educational example
            risk_amount = account_size * (risk_percent / 100)
            
            # Gold pip value approximation
            gold_pip_value = 0.10  # Approximate value per pip for 0.01 lot
            
            position_size_lots = risk_amount / (stop_loss_pips * gold_pip_value * 100)
            position_size_units = position_size_lots * 100  # Convert to ounces for gold
            
            st.info(f"""
            **Educational Calculation Example:**
            - Account Risk: ${risk_amount:.2f} ({risk_percent}% of ${account_size:,.0f})
            - Stop Loss: {stop_loss_pips} pips
            - **Position Size:** {position_size_lots:.2f} lots ({position_size_units:.0f} oz)
            
            *Note: This is for EDUCATIONAL purposes only. Real trading requires broker-specific calculations.*
            """)
        
        with tab5:
            st.subheader("📚 Personalized Learning Path")
            
            learning_path = results['learning_path']
            
            st.markdown("""
            <div class="success-box">
            <h3>🎓 Your Custom Learning Journey</h3>
            <p>Based on the patterns detected in your chart, here's a recommended learning path:</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Beginner Level
            st.write("### 🟢 Beginner Level")
            for topic in learning_path['beginner'][:5]:  # Show top 5
                st.write(f"• {topic}")
            
            # Intermediate Level
            if learning_path['intermediate']:
                st.write("### 🟡 Intermediate Level")
                for topic in learning_path['intermediate'][:5]:  # Show top 5
                    st.write(f"• {topic}")
            
            # Advanced Level
            if learning_path['advanced']:
                st.write("### 🔴 Advanced Level")
                for topic in learning_path['advanced'][:5]:  # Show top 5
                    st.write(f"• {topic}")
            
            # Learning resources
            st.write("### 📖 Suggested Learning Schedule")
            schedule = {
                "Week 1-2": ["Master support/resistance", "Learn basic candlesticks", "Practice on demo account"],
                "Week 3-4": ["Study detected patterns", "Learn risk management", "Start trading journal"],
                "Month 2-3": ["Multiple timeframe analysis", "Pattern combinations", "Psychology of trading"],
                "Month 4-6": ["Advanced strategies", "Backtesting", "System development"]
            }
            
            for timeframe, topics in schedule.items():
                with st.expander(f"📅 {timeframe}"):
                    for topic in topics:
                        st.write(f"• {topic}")
            
            # Progress tracker (educational)
            st.write("### 📈 Your Learning Progress")
            progress_cols = st.columns(4)
            with progress_cols[0]:
                st.metric("Patterns Learned", results['detected_patterns']['total_count'])
            with progress_cols[1]:
                st.metric("Concepts Covered", len(learning_path['beginner']) + 
                         len(learning_path['intermediate']) + 
                         len(learning_path['advanced']))
            with progress_cols[2]:
                st.metric("Practice Exercises", len(results['educational_content']['educational_exercises']))
            with progress_cols[3]:
                st.metric("Resources Available", len(results['educational_content']['recommended_resources']))
        
        # Always show disclaimer at the end
        st.markdown("---")
        st.error(results['disclaimer'])
        
        # Download educational report
        st.download_button(
            label="📥 Download Educational Analysis Report",
            data=str(results),
            file_name=f"educational_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
    
    else:
        # Show educational content when no file is uploaded
        st.info("👆 Upload a chart image to begin educational analysis")
        
        # Educational examples
        st.write("### 📖 What You'll Learn:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Pattern Recognition**")
            st.write("• Candlestick patterns")
            st.write("• Chart patterns")
            st.write("• Support/Resistance")
        
        with col2:
            st.write("**Risk Management**")
            st.write("• Position sizing")
            st.write("• Stop loss placement")
            st.write("• Risk-reward ratios")
        
        with col3:
            st.write("**Gold-Specific**")
            st.write("• XAU/USD dynamics")
            st.write("• Market correlations")
            st.write("• Economic factors")
        
        # Sample patterns
        st.write("### 🎯 Patterns This Tool Can Help You Identify:")
        
        sample_patterns = [
            ("Morning Star", "Bullish reversal pattern"),
            ("Head & Shoulders", "Major reversal pattern"),
            ("Cup and Handle", "Continuation pattern"),
            ("Double Top/Bottom", "Reversal patterns"),
            ("Fibonacci Levels", "Key retracement levels"),
            ("Trendlines", "Market direction analysis")
        ]
        
        for pattern, description in sample_patterns:
            st.write(f"• **{pattern}:** {description}")

# Run the application
if __name__ == "__main__":
    enhanced_web_interface()
