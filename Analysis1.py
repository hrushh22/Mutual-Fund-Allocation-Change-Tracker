import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import questionary
import sys
import glob
from rich.console import Console
from rich.table import Table
from rich import print as rprint
import re

console = Console()

class MutualFundAnalyzer:
    """Enhanced Mutual Fund Portfolio Analyzer with detailed reporting"""
    
    def __init__(self, fund_name: str = None, report_path: str = None):
        self.logger = self._setup_logging()
        self.portfolios = {}
        self.fund_name = fund_name
        self.report_path = report_path
        self.report_content = []

    def _setup_logging(self) -> logging.Logger:
        log_folder = Path('logs')
        log_folder.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_folder / 'mutual_fund_analysis.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
        
    def add_to_report(self, content: str):
        """Add content to the report"""
        self.report_content.append(content)
        
    def save_report(self):
        """Save the accumulated report content to a file"""
        if self.report_path:
            with open(self.report_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.report_content))
            self.logger.info(f"Report saved to {self.report_path}")

    def create_visualizations(self, output_dir: str):
        """Create and save visualizations"""
        try:
            # Create visualization directory
            viz_dir = Path(output_dir) / 'visualizations'
            viz_dir.mkdir(exist_ok=True)
            
            # 1. Portfolio Value Over Time
            dates = sorted(self.portfolios.keys())
            values = [self.portfolios[date]['Market value (Rs. in Lakhs)'].sum() for date in dates]
            
            plt.figure(figsize=(12, 6))
            plt.plot(dates, values, marker='o')
            plt.title('Portfolio Value Over Time')
            plt.xlabel('Date')
            plt.ylabel('Value (Rs. in Lakhs)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(viz_dir / 'portfolio_value_trend.png')
            plt.close()
            
            # 2. Top 10 Holdings for each period
            for date in dates:
                df = self.portfolios[date]
                top_10 = df.nlargest(10, '% to NAV')
                
                plt.figure(figsize=(12, 6))
                sns.barplot(data=top_10, x='% to NAV', y='Name of the Instrument')
                plt.title(f'Top 10 Holdings - {date.strftime("%B %Y")}')
                plt.xlabel('% to NAV')
                plt.tight_layout()
                plt.savefig(viz_dir / f'top_10_holdings_{date.strftime("%Y%m")}.png')
                plt.close()

            # 3. Sector Distribution Pie Chart
            for date in dates:
                df = self.portfolios[date]
                sector_data = df.groupby('Rating / Industry^')['% to NAV'].sum()

                plt.figure(figsize=(12, 8))
                plt.pie(sector_data, labels=sector_data.index, autopct='%1.1f%%')
                plt.title(f'Sector Distribution - {date.strftime("%B %Y")}')
                plt.axis('equal')
                plt.savefig(viz_dir / f'sector_distribution_{date.strftime("%Y%m")}.png')
                plt.close()
            
            # Add visualization info to report
            self.add_to_report("\n\nVisualizations have been saved in the following directory:")
            self.add_to_report(str(viz_dir))
            self.add_to_report("\nGenerated visualizations:")
            self.add_to_report("1. Portfolio Value Trend (portfolio_value_trend.png)")
            self.add_to_report("2. Top 10 Holdings for each period")
            self.add_to_report("3. Sector Distribution Pie Charts")
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {str(e)}")

    def validate_date_range(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> bool:
        if not isinstance(self.portfolios, dict):
            self.logger.error(f"Expected 'self.portfolios' to be a dictionary, but got {type(self.portfolios)}")
            return False

        available_dates = sorted(self.portfolios.keys())
        if not available_dates:
            self.logger.error("No portfolio data available to validate date range.")
            return False

        if start_date < available_dates[0] or end_date > available_dates[-1]:
            available_range = f"{available_dates[0]} to {available_dates[-1]}"
            self.logger.error(f"Date range {start_date} to {end_date} is out of bounds. Available date range is {available_range}.")
            return False

        if start_date > end_date:
            self.logger.error("Start date cannot be after end date.")
            return False

        return True

    def _extract_date_from_filename(self, file_path: str) -> pd.Timestamp:
        try:
            filename = Path(file_path).stem
            date_patterns = [
                r'\d{4}-\d{2}-\d{2}',  
                r'\w+ \d{4}'           
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, filename)
                if match:
                    return pd.to_datetime(match.group())
            
            return pd.Timestamp.now()
        except Exception as e:
            self.logger.warning(f"Could not extract date from filename: {str(e)}")
            return pd.Timestamp.now()

    def load_portfolio(self, file_path: str, sheet_name: str = 0) -> None:
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            date = self._extract_date_from_filename(file_path)
            
            header_row = None
            for idx, row in df.iterrows():
                if any('Name of the Instrument' in str(cell) for cell in row):
                    header_row = idx
                    break
            
            if header_row is None:
                raise ValueError("Could not find header row")
            
            df.columns = df.iloc[header_row]
            df = df.iloc[header_row + 1:].reset_index(drop=True)
            df.columns = [str(col).replace('\n', ' ').strip() for col in df.columns]
            
            required_columns = [
                'Name of the Instrument',
                'Market value (Rs. in Lakhs)',
                '% to NAV',
                'Rating / Industry^'
            ]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            portfolio_data = df[
                df['Name of the Instrument'].notna() & 
                ~df['Name of the Instrument'].astype(str).str.contains(
                    'EQUITY & EQUITY RELATED|Listed/awaiting listing|TOTAL|^a\)|^b\)',
                    case=False, regex=True
                )
            ].copy()
            
            portfolio_data['date'] = date
            numeric_columns = ['% to NAV', 'Market value (Rs. in Lakhs)']
            for col in numeric_columns:
                portfolio_data[col] = pd.to_numeric(portfolio_data[col], errors='coerce')
            
            self.portfolios[date] = portfolio_data
            self.logger.info(f"Successfully loaded portfolio for {date.strftime('%B %Y')}")
        except Exception as e:
            self.logger.error(f"Error loading portfolio: {str(e)}")
            raise

    def get_monthly_changes(self, start_date: pd.Timestamp, end_date: pd.Timestamp,
                            min_weight_change: float = 0.5) -> Dict:
        if not self.validate_date_range(start_date, end_date):
            raise ValueError("Invalid date range or dates not available")
        
        dates = sorted([d for d in self.portfolios.keys() if start_date <= d <= end_date])
        monthly_changes = {}
        
        for i in range(len(dates) - 1):
            current_date = dates[i]
            next_date = dates[i + 1]
            
            changes = self._analyze_changes(current_date, next_date, min_weight_change)
            period_key = f"{current_date.strftime('%B %Y')} to {next_date.strftime('%B %Y')}"
            monthly_changes[period_key] = changes
        
        return monthly_changes

    def _analyze_changes(self, start_date: pd.Timestamp, end_date: pd.Timestamp,
                     min_weight_change: float) -> Dict:
        """Analyze changes between two portfolio dates."""
        try:
            old_df = self.portfolios[start_date]
            new_df = self.portfolios[end_date]

            # Analyze holdings changes
            holdings_changes = self._analyze_holdings_changes(old_df, new_df, min_weight_change)
            
            # Calculate value changes
            old_value = old_df['Market value (Rs. in Lakhs)'].sum()
            new_value = new_df['Market value (Rs. in Lakhs)'].sum()
            value_changes = {
                'old_value': old_value,
                'new_value': new_value,
                'absolute_change': new_value - old_value,
                'percentage_change': ((new_value - old_value) / old_value * 100) if old_value != 0 else 0
            }

            # Analyze sector changes
            sector_changes = self._analyze_sector_changes(old_df, new_df)

            return {
                'holdings_changes': holdings_changes,
                'sector_changes': sector_changes,
                'value_changes': value_changes
            }
        except Exception as e:
            self.logger.error(f"Error analyzing changes: {str(e)}")
            raise

    def _analyze_sector_changes(self, old_df: pd.DataFrame, new_df: pd.DataFrame) -> Dict:
        """Analyze changes in sector allocations"""
        try:
            old_sectors = old_df.groupby('Rating / Industry^')['% to NAV'].sum()
            new_sectors = new_df.groupby('Rating / Industry^')['% to NAV'].sum()
            
            all_sectors = set(old_sectors.index) | set(new_sectors.index)
            sector_changes = {}
            
            for sector in all_sectors:
                old_weight = old_sectors.get(sector, 0)
                new_weight = new_sectors.get(sector, 0)
                
                sector_changes[sector] = {
                    'old_weight': old_weight,
                    'new_weight': new_weight,
                    'change': new_weight - old_weight
                }
            
            return sector_changes
        except Exception as e:
            self.logger.error(f"Error analyzing sector changes: {str(e)}")
            raise

    def _analyze_holdings_changes(self, old_df: pd.DataFrame, new_df: pd.DataFrame,
                                  min_weight_change: float) -> Dict:
        try:
            merged = pd.merge(
                old_df[['Name of the Instrument', '% to NAV', 'Rating / Industry^']],
                new_df[['Name of the Instrument', '% to NAV', 'Rating / Industry^']],
                on='Name of the Instrument',
                how='outer',
                suffixes=('_old', '_new')
            )

            merged = merged.fillna(0)

            new_entries = []
            exits = []
            weight_changes = []

            for _, row in merged.iterrows():
                old_weight = row['% to NAV_old'] if pd.notna(row['% to NAV_old']) else 0
                new_weight = row['% to NAV_new'] if pd.notna(row['% to NAV_new']) else 0
                weight_change = new_weight - old_weight

                if old_weight == 0 and new_weight > 0:
                    new_entries.append({
                        'instrument': row['Name of the Instrument'],
                        'weight': new_weight,
                        'sector': row['Rating / Industry^_new']
                    })
                elif old_weight > 0 and new_weight == 0:
                    exits.append({
                        'instrument': row['Name of the Instrument'],
                        'weight': old_weight,
                        'sector': row['Rating / Industry^_old']
                    })
                elif abs(weight_change) >= min_weight_change:
                    weight_changes.append({
                        'instrument': row['Name of the Instrument'],
                        'old_weight': old_weight,
                        'new_weight': new_weight,
                        'change': weight_change,
                        'sector': row['Rating / Industry^_new']
                    })

            return {
                'new_entries': new_entries,
                'exits': exits,
                'weight_changes': weight_changes
            }
        except Exception as e:
            self.logger.error(f"Error analyzing holdings changes: {str(e)}")
            raise

    def detailed_analysis_report(self, changes: Dict) -> str:
        """Generate a detailed analysis report"""
        report = []
        report.append(f"\nDetailed Analysis Report for {self.fund_name}")
        report.append("=" * 50)
        
        for period, data in changes.items():
            report.append(f"\nPeriod: {period}")
            report.append("-" * 30)
            
            # Value Changes
            value_changes = data['value_changes']
            report.append("\nPortfolio Value Changes:")
            report.append(f"Previous Value: ₹{value_changes['old_value']:,.2f} Lakhs")
            report.append(f"New Value: ₹{value_changes['new_value']:,.2f} Lakhs")
            report.append(f"Absolute Change: ₹{value_changes['absolute_change']:,.2f} Lakhs")
            report.append(f"Percentage Change: {value_changes['percentage_change']:.2f}%")
            
            # Holdings Changes
            holdings = data['holdings_changes']
            
            # New Entries
            report.append("\nNew Entries:")
            if holdings['new_entries']:
                for entry in holdings['new_entries']:
                    report.append(f"- {entry['instrument']} ({entry['sector']}) - {entry['weight']:.2f}%")
            else:
                report.append("- None")
            
            # Exits
            report.append("\nExits:")
            if holdings['exits']:
                for exit in holdings['exits']:
                    report.append(f"- {exit['instrument']} ({exit['sector']}) - {exit['weight']:.2f}%")
            else:
                report.append("- None")
            
            # Weight Changes
            report.append("\nSignificant Weight Changes:")
            if holdings['weight_changes']:
                for change in sorted(holdings['weight_changes'], 
                                   key=lambda x: abs(x['change']), 
                                   reverse=True):
                    report.append(
                        f"- {change['instrument']} ({change['sector']}): "
                        f"{change['old_weight']:.2f}% → {change['new_weight']:.2f}% "
                        f"(Δ {change['change']:+.2f}%)"
                    )
            else:
                report.append("- No significant weight changes")
            
            # Sector Changes
            report.append("\nSector Allocation Changes:")
            sector_changes = data['sector_changes']
            for sector, changes in sorted(sector_changes.items(), 
                                        key=lambda x: abs(x[1]['change']), 
                                        reverse=True):
                if abs(changes['change']) >= 0.5:  # Only show significant sector changes
                    report.append(
                        f"- {sector}: {changes['old_weight']:.2f}% → "
                        f"{changes['new_weight']:.2f}% (Δ {changes['change']:+.2f}%)"
                    )
            
            report.append("\n" + "=" * 50)
        
        return "\n".join(report)

def get_user_inputs() -> Dict:
    """Get analysis parameters from user"""
    console.print("\n[bold blue]Mutual Fund Portfolio Analysis[/bold blue]")
    
    # Fixed file paths
    file_paths = [
        r"C:\Users\hrush\Desktop\LD_tech\ZN250 - Monthly Portfolio November 2024.xlsx",
        r"C:\Users\hrush\Desktop\LD_tech\ZN250 - Monthly Portfolio September 2024.xlsx"
    ]
    
    directory = str(Path(file_paths[0]).parent)
    
    for path in file_paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"File not found: {path}")
    
    console.print("[green]Found portfolio files:[/green]")
    for file in file_paths:
        console.print(f"  - {Path(file).name}")
    
    fund_name = questionary.text("Enter the fund name:", default="ZN250").ask()
    dates = questionary.text("Enter date range (YYYY-MM-DD to YYYY-MM-DD):", default="2024-09-01 to 2024-11-01").ask()
    
    try:
        start_date, end_date = map(lambda x: pd.to_datetime(x.strip()), dates.split('to'))
    except:
        console.print("[red]Invalid date format. Using default date range.[/red]")
        start_date = pd.to_datetime("2024-09-01")
        end_date = pd.to_datetime("2024-11-30")
    
    while True:
        min_weight_change = questionary.text("Enter minimum weight change threshold (%):", default="0.1").ask()
        try:
            min_weight_change = float(min_weight_change)
            if min_weight_change < 0:
                console.print("[red]Please enter a positive number.[/red]")
                continue
            break
        except ValueError:
            console.print("[red]Please enter a valid number.[/red]")
    
    return {
        "directory": directory,
        "file_paths": file_paths,
        "fund_name": fund_name,
        "start_date": start_date,
        "end_date": end_date,
        "min_weight_change": min_weight_change
    }

def main():
    """Enhanced main execution function"""
    try:
        inputs = get_user_inputs()
        
        # Create output directory
        output_dir = Path(r"C:\Users\hrush\Desktop\LD_tech")
        output_dir.mkdir(exist_ok=True)
        
        # Create timestamped report filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = output_dir / f"{inputs['fund_name']}_analysis_report_{timestamp}.txt"
        
        analyzer = MutualFundAnalyzer(
            fund_name=inputs["fund_name"],
            report_path=str(report_path)
        )
        
        # Add initial report content
        analyzer.add_to_report(f"Mutual Fund Analysis Report - {inputs['fund_name']}")
        analyzer.add_to_report(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        analyzer.add_to_report("\nAnalysis Parameters:")
        analyzer.add_to_report(f"Date Range: {inputs['start_date'].strftime('%Y-%m-%d')} to {inputs['end_date'].strftime('%Y-%m-%d')}")
        analyzer.add_to_report(f"Minimum Weight Change Threshold: {inputs['min_weight_change']}%")
        
        console.print("\n[bold green]Loading portfolios...[/bold green]")
        for file_path in inputs["file_paths"]:
            analyzer.load_portfolio(file_path)
            analyzer.add_to_report(f"\nLoaded portfolio: {Path(file_path).name}")
        
        console.print("\n[bold green]Analyzing portfolio changes...[/bold green]")
        changes = analyzer.get_monthly_changes(
            inputs["start_date"],
            inputs["end_date"],
            inputs["min_weight_change"]
        )
        
        # Generate and add detailed analysis to report
        detailed_analysis = analyzer.detailed_analysis_report(changes)
        analyzer.add_to_report("\n" + detailed_analysis)
        
        # Create visualizations
        console.print("\n[bold green]Generating visualizations...[/bold green]")
        analyzer.create_visualizations(str(output_dir))
        
        # Save the report
        analyzer.save_report()
        
        # Print summary to console
        summary_table = Table(title="Analysis Summary")
        summary_table.add_column("Period", style="cyan")
        summary_table.add_column("New Entries", style="green")
        summary_table.add_column("Exits", style="red")
        summary_table.add_column("Value Change (%)", style="yellow")
        
        for period, data in changes.items():
            summary_table.add_row(
                period,
                str(len(data['holdings_changes']['new_entries'])),
                str(len(data['holdings_changes']['exits'])),
                f"{data['value_changes']['percentage_change']:.2f}%"
            )
        
        console.print(summary_table)
        console.print(f"\n[bold green]Analysis complete! Report saved to:[/bold green]")
        console.print(str(report_path))
        
    except Exception as e:
        console.print(f"\n[bold red]Error: {str(e)}[/bold red]")
        sys.exit(1)

if __name__ == "__main__":
    main()