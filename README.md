# IAM — Stock Market Intelligence (Flask Project)

## Project Structure

```
iam_flask/
├── server.py               ← Website server (port 8080)
├── app.py                  ← ML prediction API (port 5000)
├── requirements.txt
├── models/                 ← Place your .pkl model files here
│   ├── tcs_model.pkl
│   └── reliance_model.pkl
├── static/
│   └── new.css             ← Shared stylesheet
└── templates/              ← All HTML pages
    ├── index.html
    ├── quiz.html
    ├── holiday.html
    ├── news.html
    ├── contact.html
    ├── press_release.html
    ├── single_candlestick.html
    ├── double_candelestic.html
    ├── triple_candelestic.html
    ├── reversal_chart_pattern.html
    ├── sidewase_chart_pattern.html
    └── contiunation_chart_pattern.html
```

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your ML models
Place your `.pkl` files inside the `models/` folder.

### 3. Run the website server (Terminal 1)
```bash
python server.py
# Opens at http://localhost:8080
```

### 4. Run the ML prediction API (Terminal 2)
```bash
python app.py
# Opens at http://localhost:5000
```

## URL Routes

| URL                          | Page                        |
|------------------------------|-----------------------------|
| /                            | Home (index)                |
| /quiz                        | Stock Market Quiz           |
| /holiday                     | Market Holidays             |
| /news                        | Market News                 |
| /contact                     | Contact Us                  |
| /press                       | Press Releases              |
| /single-candlestick          | Single Candlestick Patterns |
| /double-candlestick          | Double Candlestick Patterns |
| /triple-candlestick          | Triple Candlestick Patterns |
| /reversal-patterns           | Reversal Chart Patterns     |
| /sideways-patterns           | Neutral Chart Patterns      |
| /continuation-patterns       | Continuation Chart Patterns |
