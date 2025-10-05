"""
Backtest performance analyzer.
Analyzes backtest results and generates comprehensive reports.
"""
import json
import logging
from datetime import datetime
from typing import Optional

import pandas as pd

import app.config as config
from app.db.database import SessionLocal
from app.db.models import BacktestRun, BacktestSignal

logger = logging.getLogger(__name__)


class BacktestAnalyzer:
    """Analyzes backtest results and generates performance metrics"""

    def __init__(self, run_id: int):
        """
        Initialize analyzer.

        Args:
            run_id: Backtest run ID to analyze
        """
        self.run_id = run_id
        self.db = SessionLocal()
        self.run = None
        self.signals = []

    def analyze(self, output_csv: bool = True):
        """
        Perform full analysis and generate reports.

        Args:
            output_csv: Whether to export results to CSV
        """
        # Load backtest run and signals
        self._load_data()

        if not self.run:
            logger.error(f"Backtest run {self.run_id} not found")
            return

        # Print summary to console
        self._print_console_summary()

        # Print breakdown analysis
        self._print_breakdown_analysis()

        # Print config
        self._print_config()

        # Export to CSV if requested
        if output_csv:
            self._export_to_csv()

        self.db.close()

    def _load_data(self):
        """Load backtest run and signals from database"""
        self.run = self.db.query(BacktestRun).filter(
            BacktestRun.run_id == self.run_id
        ).first()

        if not self.run:
            return

        self.signals = self.db.query(BacktestSignal).filter(
            BacktestSignal.run_id == self.run_id
        ).all()

    def _print_console_summary(self):
        """Print high-level summary to console"""
        logger.info("=" * 80)
        logger.info("BACKTEST RESULTS SUMMARY")
        logger.info("=" * 80)

        logger.info(f"Run ID: {self.run.run_id}")
        logger.info(f"Period: {self.run.start_date.date()} to {self.run.end_date.date()}")
        logger.info(f"Status: {self.run.status}")
        logger.info(f"Pairs: {json.loads(self.run.pairs)}")
        logger.info(f"Timeframes: {json.loads(self.run.timeframes)}")

        logger.info("\n--- OVERALL PERFORMANCE ---")
        logger.info(f"Total Trades: {self.run.total_trades or 0}")
        logger.info(f"Winners (TP2): {self.run.total_winners or 0}")
        logger.info(f"Partial Winners (TP1): {self.run.total_breakeven or 0}")  # TP1 stored in breakeven field
        logger.info(f"Losers (SL): {self.run.total_losers or 0}")
        logger.info(f"Win Rate: {self.run.win_rate or 0:.2f}%")
        logger.info(f"Total PnL: {self.run.total_pnl or 0:.2f}%")
        logger.info(f"Avg PnL per Trade: {self.run.avg_pnl_per_trade or 0:.2f}%")

        # Calculate additional metrics
        if self.signals:
            self._calculate_advanced_metrics()

    def _calculate_advanced_metrics(self):
        """Calculate advanced performance metrics"""
        df = pd.DataFrame([{
            'pnl': s.pnl_percent or 0.0,
            'hit': s.hit,
            'timestamp': s.hit_timestamp or s.timestamp,
            'pair': s.pair,
            'timeframe': s.timeframe,
            'side': s.side,
            'score': s.score,
            'regime': s.regime
        } for s in self.signals])

        # Max drawdown calculation
        df = df.sort_values('timestamp')
        df['cumulative_pnl'] = df['pnl'].cumsum()
        df['running_max'] = df['cumulative_pnl'].cummax()
        df['drawdown'] = df['running_max'] - df['cumulative_pnl']
        max_drawdown = df['drawdown'].max()

        # Sharpe ratio (simplified, assuming risk-free rate = 0)
        avg_return = df['pnl'].mean()
        std_return = df['pnl'].std()
        sharpe_ratio = (avg_return / std_return) if std_return > 0 else 0.0

        # Profit factor
        total_wins = df[df['pnl'] > 0]['pnl'].sum()
        total_losses = abs(df[df['pnl'] < 0]['pnl'].sum())
        profit_factor = (total_wins / total_losses) if total_losses > 0 else float('inf')

        # Average winner vs average loser
        avg_winner = df[df['pnl'] > 0]['pnl'].mean() if len(df[df['pnl'] > 0]) > 0 else 0.0
        avg_loser = df[df['pnl'] < 0]['pnl'].mean() if len(df[df['pnl'] < 0]) > 0 else 0.0

        logger.info(f"Max Drawdown: {max_drawdown:.2f}%")
        logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        logger.info(f"Profit Factor: {profit_factor:.2f}")
        logger.info(f"Avg Winner: {avg_winner:.2f}%")
        logger.info(f"Avg Loser: {avg_loser:.2f}%")

        # Update run record with these metrics
        self.run.max_drawdown = max_drawdown
        self.run.sharpe_ratio = sharpe_ratio
        self.db.commit()

    def _print_breakdown_analysis(self):
        """Print detailed breakdown by various dimensions"""
        if not self.signals:
            return

        logger.info("\n--- BREAKDOWN BY PAIR ---")
        self._breakdown_by_field('pair')

        logger.info("\n--- BREAKDOWN BY TIMEFRAME ---")
        self._breakdown_by_field('timeframe')

        logger.info("\n--- BREAKDOWN BY SIDE ---")
        self._breakdown_by_field('side')

        logger.info("\n--- BREAKDOWN BY SCORE ---")
        self._breakdown_by_score()

        logger.info("\n--- BREAKDOWN BY REGIME ---")
        self._breakdown_by_field('regime')

    def _breakdown_by_field(self, field: str):
        """
        Generate breakdown statistics by a specific field.

        Args:
            field: Field name to group by (e.g., 'pair', 'timeframe', 'side', 'regime')
        """
        df = pd.DataFrame([{
            field: getattr(s, field),
            'pnl': s.pnl_percent or 0.0,
            'hit': s.hit
        } for s in self.signals])

        grouped = df.groupby(field)

        for name, group in grouped:
            total = len(group)
            winners = len(group[group['hit'] == 'TP2'])
            losers = len(group[group['hit'] == 'SL'])
            tp1_wins = len(group[group['hit'] == 'TP1'])
            win_rate = (winners / total * 100) if total > 0 else 0.0
            total_pnl = group['pnl'].sum()
            avg_pnl = group['pnl'].mean()

            logger.info(
                f"{name:15} | Trades: {total:4} | TP2: {winners:3} | TP1: {tp1_wins:3} | SL: {losers:3} | "
                f"WR: {win_rate:5.1f}% | Total PnL: {total_pnl:7.2f}% | Avg: {avg_pnl:6.2f}%"
            )

    def _breakdown_by_score(self):
        """Breakdown by score ranges"""
        df = pd.DataFrame([{
            'score': s.score,
            'pnl': s.pnl_percent or 0.0,
            'hit': s.hit
        } for s in self.signals])

        # Create score bins
        df['score_bin'] = pd.cut(df['score'], bins=[0, 4, 5, 6, 7, 100], labels=['0-4', '5', '6', '7', '7+'])

        grouped = df.groupby('score_bin', observed=True)

        for name, group in grouped:
            total = len(group)
            winners = len(group[group['hit'] == 'TP2'])
            losers = len(group[group['hit'] == 'SL'])
            win_rate = (winners / total * 100) if total > 0 else 0.0
            total_pnl = group['pnl'].sum()
            avg_pnl = group['pnl'].mean()

            logger.info(
                f"Score {name:5} | Trades: {total:4} | W: {winners:3} | L: {losers:3} | "
                f"WR: {win_rate:5.1f}% | Total PnL: {total_pnl:7.2f}% | Avg: {avg_pnl:6.2f}%"
            )

    def _print_config(self):
        """Print config snapshot used for this backtest"""
        logger.info("\n--- BACKTEST CONFIGURATION ---")

        # Print config snapshot from run
        config_snapshot = json.loads(self.run.config_snapshot)
        for key, value in config_snapshot.items():
            logger.info(f"{key}: {value}")

        # Also print current config.py for comparison
        logger.info("\n--- CURRENT CONFIG.PY (for reference) ---")
        config_attrs = [attr for attr in dir(config) if not attr.startswith('_') and attr.isupper()]

        for attr in config_attrs[:30]:  # Print first 30 config items
            value = getattr(config, attr)
            logger.info(f"{attr}: {value}")

    def _export_to_csv(self):
        """Export backtest results to CSV"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"backtest_results_{self.run_id}_{timestamp}.csv"

        # Create DataFrame with all signal details
        data = []
        for s in self.signals:
            data.append({
                'run_id': s.run_id,
                'signal_uuid': s.signal_uuid,
                'pair': s.pair,
                'timeframe': s.timeframe,
                'side': s.side,
                'timestamp': s.timestamp,
                'price': s.price,
                'stop_loss': s.stop_loss,
                'take_profit_1': s.take_profit_1,
                'take_profit_2': s.take_profit_2,
                'hit': s.hit,
                'hit_timestamp': s.hit_timestamp,
                'hit_price': s.hit_price,
                'pnl_percent': s.pnl_percent,
                'sl_moved_to_be': s.sl_moved_to_be,
                'score': s.score,
                'required_score': s.required_score,
                'rsi': s.rsi,
                'adx': s.adx,
                'macd_diff': s.macd_diff,
                'regime': s.regime,
                'confidence': s.confidence,
                'volume_ratio': s.volume_ratio
            })

        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)

        logger.info(f"\n--- CSV EXPORT ---")
        logger.info(f"Results exported to: {filename}")


def analyze_backtest(run_id: Optional[int] = None):
    """
    Main function to analyze a backtest run.

    Args:
        run_id: Optional run ID. If not provided, analyzes the most recent run.
    """
    db = SessionLocal()

    try:
        # If no run_id provided, get the most recent
        if run_id is None:
            latest_run = db.query(BacktestRun).order_by(
                BacktestRun.created_at.desc()
            ).first()

            if not latest_run:
                logger.error("No backtest runs found in database")
                return

            run_id = latest_run.run_id
            logger.info(f"Analyzing most recent backtest run: {run_id}")

        # Create analyzer and run analysis
        analyzer = BacktestAnalyzer(run_id)
        analyzer.analyze()

    finally:
        db.close()
