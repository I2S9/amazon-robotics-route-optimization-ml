# Amazon Robotics Route Optimization with Machine Learning

A hybrid optimization and machine learning project that simulates warehouse route optimization for Amazon Robotics, combining traditional operations research (OR-Tools) with ML-based travel time prediction to improve robot routing efficiency.

## Project Overview

This project demonstrates a complete pipeline for warehouse route optimization, simulating a mini Amazon warehouse where robots must:
- Collect packages from pick-up points
- Deliver them to specific delivery stations
- Choose optimal routes to minimize travel time
- Adapt to congestion and dynamic warehouse conditions

The solution integrates:
1. **Warehouse Simulation**: 2D grid-based warehouse with robots, pick-up points, and delivery stations
2. **Route Optimization**: Google OR-Tools for solving TSP (Traveling Salesman Problem) and VRP (Vehicle Routing Problem)
3. **Machine Learning**: scikit-learn models to predict travel times based on congestion, obstacles, and warehouse conditions
4. **Hybrid Approach**: ML-enhanced optimization that uses predicted travel times instead of simple distances

## Key Features

- **2D Warehouse Simulation**: Configurable grid-based warehouse with realistic constraints
- **OR-Tools Integration**: Efficient TSP/VRP solving with multiple optimization strategies
- **ML-Based Prediction**: Random Forest and MLP models to predict travel times under various conditions
- **Comprehensive Visualizations**: Route comparisons, error analysis, and congestion heatmaps
- **Reproducible Pipeline**: Complete workflow from data generation to final results

## Repository Structure

```
amazon-robotics-route-optimization-ml/
│
├── src/
│   ├── simulate_warehouse.py      # Warehouse simulation and distance matrix generation
│   ├── optimize_routes.py          # OR-Tools route optimization (TSP/VRP)
│   ├── generate_ml_data.py        # ML dataset generation
│   └── utils.py                    # Helper functions (distances, plotting, etc.)
│
├── notebooks/
│   ├── 03_ml_data_generation.ipynb # ML dataset generation and analysis
│   ├── 04_ml_training.ipynb       # Model training and evaluation
│   └── 05_ml_integration.ipynb    # ML-enhanced optimization and visualization
│
├── data/
│   ├── raw/                        # Raw data (warehouse layout, distance matrices)
│   └── processed/                  # Processed data (optimized routes, ML models, results)
│
├── assets/                         # Screenshots and visualizations (optional)
│
├── report/                         # Project report (optional)
│
├── run_all_notebooks.py           # Script to execute all notebooks in order
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd amazon-robotics-route-optimization-ml
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python -c "import ortools, sklearn, numpy, pandas, matplotlib; print('[OK] All dependencies installed')"
   ```

## Usage

### Quick Start

1. **Generate warehouse simulation**:
   ```bash
   python src/simulate_warehouse.py
   ```
   This creates:
   - Warehouse layout with robots, pick-up points, and delivery stations
   - Distance matrix (Euclidean or Manhattan)
   - Visualization saved to `data/raw/warehouse_layout.png`

2. **Run route optimization (baseline)**:
   ```bash
   python src/optimize_routes.py
   ```
   This solves TSP/VRP using OR-Tools and saves:
   - Optimized routes CSV
   - Route visualization
   - Optimization metadata

3. **Execute ML pipeline** (recommended):
   ```bash
   python run_all_notebooks.py
   ```
   This executes all notebooks in order:
   - `03_ml_data_generation.ipynb`: Generate ML dataset
   - `04_ml_training.ipynb`: Train ML models
   - `05_ml_integration.ipynb`: Integrate ML with optimization

   Alternatively, run notebooks individually in Jupyter.

### Manual Execution

For step-by-step execution:

1. **Generate ML dataset**:
   - Open `notebooks/03_ml_data_generation.ipynb`
   - Run all cells to generate synthetic dataset with congestion, obstacles, and travel times

2. **Train ML models**:
   - Open `notebooks/04_ml_training.ipynb`
   - Run all cells to train Random Forest and MLP models
   - Models are saved to `data/processed/ml_model_rf.pkl`

3. **ML-enhanced optimization**:
   - Open `notebooks/05_ml_integration.ipynb`
   - Run all cells to:
     - Build ML-based cost matrix
     - Solve optimization with ML predictions
     - Compare baseline vs ML routes
     - Generate comprehensive visualizations

## Results and Performance

### ML Model Performance

- **Random Forest Regressor**:
  - RMSE: ~15-25% of mean travel time
  - MAE: ~10-20% of mean travel time
  - R²: 0.75-0.90

- **MLP Regressor**:
  - Comparable performance to Random Forest
  - Better generalization on unseen congestion patterns

### Optimization Results

- **Baseline (Distance-based)**:
  - Uses simple Euclidean/Manhattan distances
  - Fast computation
  - May not account for congestion

- **ML-Enhanced**:
  - Uses predicted travel times considering congestion and obstacles
  - Routes adapt to warehouse conditions
  - Typically 5-15% improvement in congested scenarios
  - Equivalent or better performance in normal conditions

### Visualizations

The project generates several visualizations:

1. **Warehouse Layout**: Initial warehouse configuration with all points
2. **Route Comparison**: Side-by-side comparison of baseline vs ML routes
3. **ML Error Analysis**: 4-panel plot showing prediction accuracy
4. **Congestion Heatmaps**: Warehouse congestion distribution with route overlays

All visualizations are saved to `data/processed/`.

## Technical Details

### Technologies Used

- **Python 3.10+**: Core programming language
- **OR-Tools**: Google's optimization library for TSP/VRP solving
- **scikit-learn**: Machine learning models (RandomForest, MLP)
- **NumPy & Pandas**: Data manipulation and numerical computing
- **Matplotlib & Seaborn**: Data visualization
- **Jupyter Notebooks**: Interactive analysis and documentation

### Key Algorithms

1. **TSP/VRP Solving**:
   - OR-Tools Routing Solver
   - Search strategies: PATH_CHEAPEST_ARC, GUIDED_LOCAL_SEARCH
   - Time limits: 30 seconds (configurable)

2. **ML Models**:
   - Random Forest Regressor (100 trees, max_depth=10)
   - MLP Regressor (2 hidden layers, 50 neurons each)
   - Features: coordinates, distance, congestion, obstacles, robot count

3. **Distance Metrics**:
   - Euclidean distance (default)
   - Manhattan distance (optional)

## Project Objectives

This project demonstrates:

1. **Operations Research**: Application of TSP/VRP algorithms to warehouse logistics
2. **Machine Learning**: Predictive modeling for travel time estimation
3. **Hybrid Systems**: Combining traditional optimization with ML predictions
4. **Real-world Application**: Simulation of Amazon Robotics warehouse scenarios
5. **End-to-end Pipeline**: Complete workflow from data generation to visualization

## Performance Metrics

- **Dataset Size**: 1,000-10,000 samples (configurable)
- **Warehouse Grid**: 20×20 (configurable)
- **Optimization Time**: <30 seconds for typical problems
- **ML Training Time**: <5 minutes for 10K samples
- **Model Accuracy**: R² > 0.75, MAE < 20% of mean

## Future Improvements

- Real-time congestion updates
- Multi-objective optimization (time, energy, priority)
- Reinforcement learning for adaptive routing
- Integration with actual warehouse data
- Distributed optimization for large-scale warehouses

## Contributing

This is a portfolio project for an Applied Scientist Internship position. For questions or feedback, please contact the repository owner.

## License

This project is for educational and portfolio purposes.

## Acknowledgments

- Google OR-Tools for optimization algorithms
- scikit-learn team for ML tools
- Amazon Robotics for inspiration and real-world context

---

**Note**: This project simulates warehouse operations and is not connected to actual Amazon systems. All data is synthetic and generated for demonstration purposes.

