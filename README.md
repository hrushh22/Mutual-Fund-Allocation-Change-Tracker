# 📊 Mutual Fund Allocation Change Tracker

> A robust Python framework for analyzing and tracking mutual fund allocation changes over time.

## 🎯 Features

- 📈 Track mutual fund allocation changes over time
- 🔍 Detailed portfolio analysis with customizable parameters
- 📊 Multiple visualization options (trends, holdings, sectors)
- 📝 Comprehensive reporting system
- 🚀 Interactive command-line interface

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mutual-fund-tracker.git
cd mutual-fund-tracker
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:

## 📋 Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- plotly
- rich
- questionary

## 🚀 Quick Start

1. Prepare your data files:
   - Excel files with mutual fund portfolio data
   - Required columns: 'Name of the Instrument', 'Market value', '% to NAV', 'Rating / Industry'

2. Run the analyzer:
```bash
python Analysis1.py
```

3. Follow the interactive prompts to:
   - Enter fund name
   - Specify date range
   - Set minimum weight change threshold

## 💡 Usage Example

```python
from mutual_fund_analyzer import MutualFundAnalyzer

# Initialize analyzer
analyzer = MutualFundAnalyzer(
    fund_name="ZN250",
    report_path="output/analysis_report.txt"
)

# Load portfolio data
analyzer.load_portfolio("path/to/portfolio.xlsx")

# Generate analysis
changes = analyzer.get_monthly_changes(
    start_date="2024-09-01",
    end_date="2024-11-01",
    min_weight_change=0.5
)

# Create visualizations
analyzer.create_visualizations("output/")
```

## 📊 Output Examples

The analyzer generates several types of outputs:

### 1. Portfolio Value Trends
```
📈 Tracks changes in total portfolio value over time
```

### 2. Holdings Analysis
```
➕ New Entries
➖ Exits
🔄 Weight Changes
```

### 3. Sector Distribution
```
🔸 Sector-wise allocation
🔸 Sector-wise changes
🔸 Visual distribution charts
```

## 📝 Configuration

Key parameters that can be configured:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `min_weight_change` | Minimum % change to track | 0.5 |
| `report_path` | Output report location | `./output/report.txt` |
| `log_level` | Logging detail level | `INFO` |

## 🔍 Features in Detail

### Portfolio Analysis
- 📊 Track value changes over time
- 🔄 Monitor holding changes
- 📈 Analyze sector rotations
- 📉 Identify significant weight changes

### Reporting
- 📑 Detailed text reports
- 📊 Visual charts and graphs
- 📋 Summary statistics
- 💾 Data export capabilities

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Mutual fund data providers
- Python financial community
- Open-source contributors

## ⭐ Support

If you find this project useful, please consider giving it a star on GitHub!

---
Made with ❤️ by Hrushik Mehta
