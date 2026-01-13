# Kenyan Animal Disease Prediction System

A machine learning-powered system for predicting and managing livestock and poultry diseases in Kenya. This system uses advanced algorithms to analyze animal health data and provide accurate disease predictions with context-specific recommendations.

##  Features

- **Disease Prediction**: Predict diseases for individual animals or process batch data
- **Multi-Animal Support**: Specialized models for livestock (cattle, goats, sheep) and poultry (chickens, turkeys, ducks)
- **Kenyan Context**: Tailored for Kenyan agricultural conditions, diseases, and treatment guidelines
- **Dual Interface**: Command-line interface and REST API for flexible integration
- **Model Training**: Train custom models on your historical data
- **Data Validation**: Comprehensive data quality checks and preprocessing
- **Disease Information**: Detailed information about diseases, symptoms, and treatments
- **Sample Data Generation**: Generate synthetic data for testing and development
- **Prediction Explanations**: Understand why the model made specific predictions

##  Architecture

The system is organized into modular components:

```
├── main.py                 # Command-line interface
├── api/endpoints.py        # FastAPI REST endpoints
├── models/                 # Machine learning models
│   ├── base_animal_model.py
│   ├── livestock_model.py  # RandomForest for livestock
│   ├── poultry_model.py    # Gradient Boosting for poultry
│   └── model_registry.py
├── config/                 # Configuration and disease mappings
│   ├── animal_configs.py
│   └── disease_mappings.py
├── preprocessing/          # Data preprocessing pipeline
├── data_handlers/          # Data loading and processing
└── utils/                  # Utility functions
```

##  Supported Diseases

### Livestock Diseases
- **East Coast Fever** (Nagana ya Pwani) - Cattle
- **Foot and Mouth Disease** (Ugonjwa wa Mdomo na Magufuli) - Cattle, goats, sheep
- **Mastitis** (Ugonjwa wa Titi) - Dairy cattle
- **Lumpy Skin Disease** (Ugonjwa wa Ngozi) - Cattle

### Poultry Diseases
- **Newcastle Disease** (Ugoni wa Kuku) - Chickens, turkeys
- **Infectious Bursal Disease (Gumboro)** - Broilers, layers
- **Fowl Typhoid** (Homa ya Kuku) - Layers, broilers
- **Avian Influenza** (Homa ya Ndege) - All poultry types

##  Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

##  Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd machine_learning_classification_RandomForest
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python main.py --mode info
```

##  Usage

### Command Line Interface

The system provides a comprehensive CLI with multiple modes:

#### Get System Information
```bash
python main.py --mode info
```

#### Train a Model
```bash
# Train livestock model
python main.py --mode train --animal-type livestock --data-file your_livestock_data.csv

# Train poultry model
python main.py --mode train --animal-type poultry --data-file your_poultry_data.csv
```

#### Make Predictions

**Single Prediction:**
```bash
# Livestock prediction
python main.py --mode predict --animal-type livestock --input '{"animal_type": "dairy_cattle", "age_months": 36, "body_temperature": 40.5, "feed_intake": "reduced", "county": "Nakuru", "season": "long_rains"}'

# Poultry prediction
python main.py --mode predict --animal-type poultry --input '{"poultry_type": "layers", "age_weeks": 32, "mortality_rate": 2.5, "county": "Kiambu", "season": "cold"}'
```

**Batch Prediction:**
```bash
python main.py --mode batch --animal-type livestock --data-file batch_cases.csv
```

#### Validate Data
```bash
python main.py --mode validate --animal-type livestock --data-file data_to_validate.csv
```

#### Get Disease Information
```bash
# List all diseases
python main.py --mode diseases --animal-type livestock

# Get specific disease info
python main.py --mode diseases --animal-type livestock --disease east_coast_fever
```

#### Generate Sample Data
```bash
python main.py --mode sample --animal-type poultry --n-samples 100 --save-path sample_poultry.csv
```

### REST API

Start the FastAPI server:
```bash
python -m uvicorn api.endpoints:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

#### API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `GET /info` - System information
- `GET /models` - List available models
- `POST /predict/livestock` - Predict livestock disease
- `POST /predict/poultry` - Predict poultry disease
- `POST /predict/batch` - Batch prediction from CSV
- `POST /train` - Train/re-train model
- `GET /diseases` - Get disease information
- `POST /generate-sample-data` - Generate sample data

#### Example API Usage

**Predict Livestock Disease:**
```bash
curl -X POST "http://localhost:8000/predict/livestock" \
     -H "Content-Type: application/json" \
     -d '{
       "animal_type": "dairy_cattle",
       "age_months": 36,
       "body_temperature": 40.5,
       "feed_intake": "reduced",
       "water_intake": "decreased",
       "milk_production": 8.5,
       "county": "Nakuru",
       "season": "long_rains",
       "vaccination_status": "partial"
     }'
```

**Batch Prediction:**
```bash
curl -X POST "http://localhost:8000/predict/batch?animal_type=livestock" \
     -F "file=@batch_data.csv"
```

##  Data Format

### Livestock Data Columns
- `animal_type`: Type of livestock (dairy_cattle, beef_cattle, goats, sheep)
- `age_months`: Age in months (1-240)
- `body_temperature`: Temperature in Celsius (35.0-42.0)
- `feed_intake`: Feed intake status (normal, reduced, very_low)
- `water_intake`: Water intake status
- `milk_production`: Milk production in liters/day (0 for non-dairy)
- `county`: Kenyan county
- `season`: Current season (long_rains, short_rains, dry_season, cold_season)
- `disease_diagnosis`: Target column (disease name)

### Poultry Data Columns
- `poultry_type`: Type of poultry (layers, broilers, local_chickens, turkeys, ducks)
- `age_weeks`: Age in weeks (1-100)
- `flock_size`: Number of birds in flock
- `mortality_rate`: Mortality rate percentage (0-100)
- `egg_production`: Egg production percentage (0-100, 0 for non-layers)
- `feed_consumption`: Daily feed consumption in kg
- `county`: Kenyan county
- `season`: Current season
- `disease_diagnosis`: Target column

## 🔧 Configuration

The system is highly configurable through the `config/` directory:

- `animal_configs.py`: Model configurations, disease mappings, validation rules
- `disease_mappings.py`: Symptom-disease relationships

##  Model Performance

### Livestock Model (RandomForest)
- **Algorithm**: Random Forest Classifier
- **Hyperparameters**: 200 estimators, max_depth=15, class_weight=balanced
- **Features**: 15+ features including animal type, age, temperature, location
- **Expected Accuracy**: 85-95% (depending on data quality)

### Poultry Model (Gradient Boosting)
- **Algorithm**: Gradient Boosting Classifier
- **Hyperparameters**: 150 estimators, learning_rate=0.1, max_depth=10
- **Features**: 12+ features including poultry type, flock metrics, production data
- **Expected Accuracy**: 88-96% (depending on data quality)

##  Disease Management Features

For each predicted disease, the system provides:

- **Immediate Actions**: Step-by-step response guidelines
- **Treatment Protocols**: Kenyan-specific treatment recommendations
- **Prevention Measures**: Long-term prevention strategies
- **Reporting Requirements**: Legal reporting obligations
- **Economic Impact Assessment**: Cost estimates and production loss analysis
- **Risk Assessment**: Containment urgency and risk levels

##  Testing

### Generate Test Data
```bash
# Generate sample livestock data
python main.py --mode sample --animal-type livestock --n-samples 1000 --save-path test_livestock.csv

# Generate sample poultry data
python main.py --mode sample --animal-type poultry --n-samples 1000 --save-path test_poultry.csv
```

### Run Validation Tests
```bash
python main.py --mode validate --animal-type livestock --data-file test_livestock.csv
```

##  Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

##  License

This project is licensed under the MIT License - see the LICENSE file for details.

##  Support

For support and questions:
- **Email**: support@animaldisease.co.ke
- **Helpline**: 0800-722-001 (Kenya Vet Helpline)
- **Documentation**: [Full API Documentation](http://localhost:8000/docs) (when running)

##  Acknowledgments

- Developed for the Kenyan agricultural sector
- Uses local disease nomenclature and treatment guidelines
- Incorporates seasonal and regional risk factors specific to Kenya
- Built with modern machine learning practices and FastAPI framework

---

**Version**: 1.0.0
**Last Updated**: 2024
**Authors**: Animal Disease Prediction Team