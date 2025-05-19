# Solar Challenge Week 1

This repository contains the analysis of solar farm data from Benin, Sierra Leone, and Togo for MoonLight Energy Solutions. The project aims to identify high-potential regions for solar installation through data analysis and visualization.

## Project Structure

```
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows
│       ├── ci.yml
├── .gitignore
├── requirements.txt
├── README.md
├── src/
├── notebooks/
│   ├── __init__.py
│   └── README.md
├── tests/
│   ├── __init__.py
└── scripts/
    ├── __init__.py
    └── README.md
├── app/
    ├── __init__.py
    ├── main.py
    └── utils.py
```

## Environment Setup

### Prerequisites
- Python 3.8 or higher
- Git

### Clone the Repository

```bash
git clone https://github.com/yourusername/solar-challenge-week1.git
cd solar-challenge-week1
```

### Set Up Virtual Environment

#### Using venv
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### Using conda
```bash
conda create -n solar-env python=3.10
conda activate solar-env
pip install -r requirements.txt
```

## Data

The data for this project includes solar radiation measurements from Benin, Sierra Leone, and Togo. The data files should be placed in a `data/` directory (which is ignored by git).

## Running the Analysis

The analysis notebooks can be found in the `notebooks/` directory. To run them:

```bash
jupyter notebook notebooks/
```

## Dashboard

The Streamlit dashboard can be run with:

```bash
streamlit run app/main.py
```

## Tests

To run the tests:

```bash
pytest tests/
```

## Contributing

1. Create a new branch for your feature
2. Make your changes
3. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
