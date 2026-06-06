name: ML CorrArb Pipeline

on:
  workflow_dispatch: 
    inputs:
      max_splits:
        description: 'Maximum Walk-Forward Splits (Leave empty for all)'
        required: false
        default: ''

jobs:
  run-backtest-and-train:
    runs-on: ubuntu-latest
    timeout-minutes: 360 # حداکثر زمان گیت‌هاب اکشنز (۶ ساعت)

    steps:
      - name: Checkout Repository
        uses: actions/checkout@v4

      - name: Setup Python 3.10
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'
          cache: 'pip'

      - name: Install Dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Run ML Pipeline
        run: |
          if [ -z "${{ github.event.inputs.max_splits }}" ]; then
            python main_ml.py --data data --output ml_models
          else
            python main_ml.py --data data --output ml_models --max_splits ${{ github.event.inputs.max_splits }}
          fi

      - name: Upload Models and Reports
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: ML_Models_and_Results
          path: ml_models/
          retention-days: 14
