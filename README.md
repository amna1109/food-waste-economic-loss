# Food Waste Economic Loss Prediction (UNEP)
My final project on global food waste management for UNEP.
## The problem
Countries measure waste in tons, so it's harder to identify how much money was lost. We decided to come up with a solution to turn the tons into dollars so that UNEP can see which country has the highest predicted economic loss, as well as which food category is costing the country.
## Team
- Amna Saood – ID:H00535107
- Maryam Yousef – ID:H00535133

## Project structure
- `data/` – raw dataset (Excel) and cleaned dataset (CSV)
- `docs/` – data dictionary, quality reports before/after cleaning, cleaning decisions
- `notebooks/01_data_cleaning.ipynb` – data understanding and cleaning (Stages 1–3)
- `requirements.txt` – Python packages needed

## How to run
1. Install the packages: `pip install -r requirements.txt`
2. Open `notebooks/01_data_cleaning.ipynb` and run all cells.

## AI use
We used AI tools CALUDE, DEEPSEEK AND CHATGPT to help generate the code and to run a quick analysis to check whether our choice of prediction and model was possible. All written explanations are our own.

## Live app
https://food-waste-unep.streamlit.app/

## How to run the app locally
python -m streamlit run app/app.py
