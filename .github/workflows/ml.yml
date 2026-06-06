name: ML CorrArb - Parallel Training

on:
  workflow_dispatch: 

jobs:
  train-splits:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false # اگر یک ماشین ارور داد، بقیه متوقف نشوند
      matrix:
        # برای 16 سال داده، حدود 20 اسپلیت خواهیم داشت. (از 0 تا 19)
        split: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

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

      - name: Run Model Training for Split ${{ matrix.split }}
        run: |
          python main_ml.py --data data --output ml_models_${{ matrix.split }} --split_idx ${{ matrix.split }}

      - name: Upload Split ${{ matrix.split }} Artifacts
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: Model_Split_${{ matrix.split }}
          path: ml_models_${{ matrix.split }}/
          retention-days: 10
