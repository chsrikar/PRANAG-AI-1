"""
dashboard_generator.py — DIVYANSHU (Additional Component)
Generates interactive HTML dashboard visualizing validation results across 5 tabs.
"""

import json
import os
from pathlib import Path


def generate_dashboard(validation_report_path: str, output_path: str = "validation_dashboard.html"):
    # Load validation report
    with open(validation_report_path, "r") as f:
        report = json.load(f)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PRANAG-AI Validation Dashboard</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }}
        .tabs {{
            display: flex;
            background: #f8f9fa;
            border-bottom: 1px solid #dee2e6;
        }}
        .tab {{
            flex: 1;
            padding: 15px;
            text-align: center;
            cursor: pointer;
            border-bottom: 3px solid transparent;
            transition: all 0.3s;
        }}
        .tab.active {{
            background: white;
            border-bottom-color: #667eea;
            color: #667eea;
            font-weight: bold;
        }}
        .tab-content {{
            display: none;
            padding: 20px;
        }}
        .tab-content.active {{
            display: block;
        }}
        .metric-card {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #667eea;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .chart-container {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
        }}
        .progress-bar {{
            background: #e9ecef;
            border-radius: 4px;
            height: 20px;
            margin: 5px 0;
            overflow: hidden;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #28a745, #20c997);
            transition: width 0.3s;
        }}
        .status-good {{ color: #28a745; font-weight: bold; }}
        .status-warning {{ color: #ffc107; font-weight: bold; }}
        .status-bad {{ color: #dc3545; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 PRANAG-AI Validation Dashboard</h1>
            <p>Report ID: {report['report_id']} | Generated: {report['generated_at'][:19]}</p>
        </div>

        <div class="tabs">
            <div class="tab active" onclick="showTab(0)">Overview</div>
            <div class="tab" onclick="showTab(1)">Designs</div>
            <div class="tab" onclick="showTab(2)">Accuracy</div>
            <div class="tab" onclick="showTab(3)">Failures</div>
            <div class="tab" onclick="showTab(4)">Docs</div>
        </div>

        <!-- Overview Tab -->
        <div class="tab-content active">
            <h2>📊 Executive Summary</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <h3>Total Designs</h3>
                    <p style="font-size: 2em; margin: 0;">{report['executive_summary']['total_designs']}</p>
                </div>
                <div class="metric-card">
                    <h3>Pass Rate</h3>
                    <p style="font-size: 2em; margin: 0;">{report['executive_summary']['pass_rate_pct']:.1f}%</p>
                </div>
                <div class="metric-card">
                    <h3>Surrogate Accuracy</h3>
                    <p style="font-size: 2em; margin: 0;">{report['executive_summary']['surrogate_accuracy']}</p>
                </div>
                <div class="metric-card">
                    <h3>System Status</h3>
                    <p style="font-size: 1.2em; margin: 0;" class="{'status-good' if report['executive_summary']['system_status'] == 'OPERATIONAL' else 'status-warning'}">{report['executive_summary']['system_status']}</p>
                </div>
            </div>

            <h3>🎯 Top Recommendations</h3>
            <ul>
                {"".join(f"<li>{rec}</li>" for rec in report['recommendations'])}
            </ul>
        </div>

        <!-- Designs Tab -->
        <div class="tab-content">
            <h2>🏆 Top Designs</h2>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Design ID</th>
                        <th>Score</th>
                        <th>Biology</th>
                        <th>Materials</th>
                        <th>Physics</th>
                        <th>Chemistry</th>
                        <th>Recommendation</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(f'''
                    <tr>
                        <td>{d['rank']}</td>
                        <td>{d['design_id']}</td>
                        <td>{d['score']:.4f}</td>
                        <td>{d['biology_score']:.4f}</td>
                        <td>{d['materials_score']:.4f}</td>
                        <td>{d['physics_score']:.4f}</td>
                        <td>{d['chemistry_score']:.4f}</td>
                        <td>{d['recommendation']}</td>
                    </tr>
                    ''' for d in report['top_designs'])}
                </tbody>
            </table>
        </div>

        <!-- Accuracy Tab -->
        <div class="tab-content">
            <h2>🎯 Accuracy Analysis</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <h3>Accuracy</h3>
                    <p style="font-size: 2em; margin: 0;">{report['accuracy_summary']['accuracy_pct']:.1f}%</p>
                </div>
                <div class="metric-card">
                    <h3>False Positive Rate</h3>
                    <p style="font-size: 2em; margin: 0;">{report['accuracy_summary']['false_positive_pct']:.1f}%</p>
                </div>
                <div class="metric-card">
                    <h3>False Negative Rate</h3>
                    <p style="font-size: 2em; margin: 0;">{report['accuracy_summary']['false_negative_pct']:.1f}%</p>
                </div>
                <div class="metric-card">
                    <h3>F1 Score</h3>
                    <p style="font-size: 2em; margin: 0;">{report['accuracy_summary']['f1_score']:.4f}</p>
                </div>
            </div>

            <h3>Confusion Matrix</h3>
            <table style="max-width: 400px;">
                <thead>
                    <tr>
                        <th></th>
                        <th>Predicted Pass</th>
                        <th>Predicted Fail</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Actual Pass</strong></td>
                        <td>{report['accuracy_summary'].get('confusion_matrix', {}).get('TP', 'N/A')}</td>
                        <td>{report['accuracy_summary'].get('confusion_matrix', {}).get('FN', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td><strong>Actual Fail</strong></td>
                        <td>{report['accuracy_summary'].get('confusion_matrix', {}).get('FP', 'N/A')}</td>
                        <td>{report['accuracy_summary'].get('confusion_matrix', {}).get('TN', 'N/A')}</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <!-- Failures Tab -->
        <div class="tab-content">
            <h2>❌ Failure Analysis</h2>
            <div class="metric-card">
                <h3>Failure Distribution</h3>
                {"".join(f'''
                <div>
                    <strong>{cat.replace('_', ' ').title()}:</strong> {pct:.1f}%
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {pct}%"></div>
                    </div>
                </div>
                ''' for cat, pct in report['failure_summary']['distribution'].items() if pct > 0)}
            </div>

            <h3>Severity Breakdown</h3>
            <div class="metric-grid">
                {"".join(f'''
                <div class="metric-card">
                    <h3>{severity.title()}</h3>
                    <p style="font-size: 2em; margin: 0;">{count}</p>
                </div>
                ''' for severity, count in report['failure_summary']['severity'].items())}
            </div>
        </div>

        <!-- Docs Tab -->
        <div class="tab-content">
            <h2>📚 Documentation</h2>
            <h3>PRANAG-AI Validation Pipeline</h3>
            <p>This dashboard visualizes the results of the PRANAG-AI material design validation system.</p>

            <h4>Components:</h4>
            <ul>
                <li><strong>Failure Analyzer:</strong> Classifies failures into 5 categories with actionable suggestions</li>
                <li><strong>FP Fixer:</strong> Sweeps thresholds 0.60–0.85 to find optimal false positive rate below 5%</li>
                <li><strong>Feedback Loop:</strong> Logs failed designs and suggests threshold adjustments</li>
                <li><strong>Accuracy Validator:</strong> Compares surrogate vs full physics with calibration</li>
                <li><strong>Cross-Domain Validator:</strong> Ensures consistency across biology, chemistry, physics, materials</li>
            </ul>

            <h4>Thresholds:</h4>
            <ul>
                <li>Pass Threshold: 0.70</li>
                <li>Accuracy Target: >95%</li>
                <li>False Positive Max: <5%</li>
                <li>False Negative Max: <10%</li>
            </ul>
        </div>
    </div>

    <script>
        function showTab(tabIndex) {{
            const tabs = document.querySelectorAll('.tab');
            const contents = document.querySelectorAll('.tab-content');

            tabs.forEach((tab, i) => {{
                if (i === tabIndex) {{
                    tab.classList.add('active');
                }} else {{
                    tab.classList.remove('active');
                }}
            }});

            contents.forEach((content, i) => {{
                if (i === tabIndex) {{
                    content.classList.add('active');
                }} else {{
                    content.classList.remove('active');
                }}
            }});
        }}
    </script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"📊 Dashboard generated: {output_path}")
    return output_path


if __name__ == "__main__":
    report_path = "results/validation_report.json"
    if os.path.exists(report_path):
        generate_dashboard(report_path)
    else:
        print("❌ Validation report not found. Run validation first.")