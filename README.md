
# Sales Forecast App

It is a web application for predicting sales volume using machine learning.

## Requirements

- Python 3.9+
- Docker (optional)

## Installation

### Local installation

1. Clone the repository:
   ```bash
   git clone https://github.com/nelezatovrezhu/sales-forecast-app.git
   cd sales-forecast-app

2. Install the dependencies:
    pip install -r requirements.txt

3. Launch the app:
    python api/app.py

The application will be available at: http://127.0.0.1:5000/

#### Using Docker

1. Build a Docker image:

    docker build -t sales-forecast-app .
2. Launch the container:

    docker run -d -p 5000:5000 --name sales-forecast-container sales-forecast-app

The application will be available at: http://127.0.0.1:5000/

How to use
Upload historical data in CSV format (sample file: example.csv).
Click the "Train" button
Select a forecast scenario or enter your own data.
Click the "Predict" button to get a forecast.

