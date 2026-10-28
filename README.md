# CricReplay — Cricket What-If Lab

CricReplay is an IPL cricket analytics project that explores historical bowling performance and experimental match scenarios using ball-by-ball data.

## Features

- Historical bowling analysis by match phase.
- Bowler comparison using historical performance.
- Over-level what-if analysis.
- Experimental full-match impact simulations.
- Interactive Streamlit dashboard with charts and downloadable results.

## Technology Stack

- Python
- Pandas and NumPy
- Streamlit
- Altair
- Cricsheet IPL ball-by-ball data

## Run Locally

1. Clone this repository.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the IPL data from [Cricsheet](https://cricsheet.org/downloads/) and run the project's data-processing script if the processed files are not available locally.
4. Place the required data files in the expected `Data/` folders.
5. Launch the dashboard:

   ```bash
   streamlit run app.py
   ```

## Data and Limitations

The datasets are excluded from this repository and must be obtained separately. Please follow Cricsheet's current terms and attribution requirements.

Simulation results are exploratory estimates based on historical data. They are not guaranteed predictions of real match outcomes.

## Project Status

The dashboard, data validation and a real-data match-impact simulation have been tested locally. Further validation of model accuracy and edge cases remains future work.
