# Probabilistic Intraday Trading Forecast System - Architecture Diagrams

This document contains comprehensive Mermaid diagrams illustrating the system architecture, data flow, model design, and deployment structure.

## 1. High-Level System Architecture

```mermaid
graph TB
    subgraph "Data Sources"
        A1[Alpaca API<br/>OHLCV + VWAP]
        A2[Yahoo Finance<br/>Historical Data]
        A3[News APIs<br/>Financial Headlines]
    end
    
    subgraph "Data Collection Layer"
        B1[Data Fetcher<br/>Minute-level Data]
        B2[Data Validator<br/>Quality Checks]
        B3[Data Storage<br/>Parquet/CSV]
    end
    
    subgraph "Feature Engineering Layer"
        C1[Technical Indicators<br/>MACD, RSI, BB, ATR]
        C2[Sentiment Analysis<br/>FinBERT Model]
        C3[Feature Merger<br/>OHLCV + Indicators + Sentiment]
    end
    
    subgraph "Model Architecture"
        D1[LSTM Layers<br/>Short-term Dependencies]
        D2[Transformer Encoder<br/>Long-range Attention]
        D3[Feature Fusion<br/>Combine Outputs]
        D4[Bayesian Head<br/>Monte Carlo Dropout]
    end
    
    subgraph "Prediction Output"
        E1[Direction Prediction<br/>Up/Down Probability]
        E2[Level-Reach Probability<br/>Today's Open, Yesterday's Open/Close]
        E3[Confidence Intervals<br/>Uncertainty Quantification]
    end
    
    subgraph "Deployment Layer"
        F1[FastAPI/Flask<br/>REST API]
        F2[TradingView<br/>Pine Script Integration]
        F3[Dash Dashboard<br/>Real-time Visualization]
    end
    
    A1 --> B1
    A2 --> B1
    A3 --> B1
    B1 --> B2
    B2 --> B3
    B3 --> C1
    B3 --> C2
    C1 --> C3
    C2 --> C3
    C3 --> D1
    C3 --> D2
    D1 --> D3
    D2 --> D3
    D3 --> D4
    D4 --> E1
    D4 --> E2
    D4 --> E3
    E1 --> F1
    E2 --> F1
    E3 --> F1
    F1 --> F2
    F1 --> F3
```

## 2. Detailed Model Architecture

```mermaid
graph LR
    subgraph "Input Features"
        I1[OHLCV Data<br/>60 time steps]
        I2[Technical Indicators<br/>MACD, RSI, BB, ATR]
        I3[Sentiment Scores<br/>FinBERT Output]
    end
    
    subgraph "LSTM Branch"
        L1[LSTM Layer 1<br/>128 units]
        L2[LSTM Layer 2<br/>64 units]
        L3[Dropout 0.3]
        L4[LSTM Output<br/>Short-term Features]
    end
    
    subgraph "Transformer Branch"
        T1[Positional Encoding]
        T2[Multi-Head Attention<br/>8 heads]
        T3[Feed Forward<br/>512 units]
        T4[Layer Norm]
        T5[Transformer Output<br/>Long-range Features]
    end
    
    subgraph "Fusion & Probabilistic"
        F1[Concatenate<br/>LSTM + Transformer]
        F2[Dense Layer<br/>256 units]
        F3[Dropout 0.5]
        F4[Monte Carlo Dropout<br/>Bayesian Inference]
        F5[Output Head 1<br/>Direction Classification]
        F6[Output Head 2<br/>Level-Reach Probability]
    end
    
    I1 --> L1
    I2 --> L1
    I3 --> L1
    I1 --> T1
    I2 --> T1
    I3 --> T1
    
    L1 --> L2
    L2 --> L3
    L3 --> L4
    
    T1 --> T2
    T2 --> T3
    T3 --> T4
    T4 --> T5
    
    L4 --> F1
    T5 --> F1
    F1 --> F2
    F2 --> F3
    F3 --> F4
    F4 --> F5
    F4 --> F6
```

## 3. Data Flow Architecture

```mermaid
flowchart TD
    Start([Market Opens]) --> Collect[Collect Real-time Data]
    
    Collect --> Alpaca[Alpaca API<br/>Minute-level OHLCV]
    Collect --> News[News APIs<br/>Financial Headlines]
    
    Alpaca --> Validate[Data Validation<br/>Outlier Detection ±3σ]
    News --> Sentiment[Sentiment Analysis<br/>FinBERT Model]
    
    Validate --> Clean[Data Cleaning<br/>Handle Missing Values]
    Clean --> Normalize[Z-score Normalization]
    
    Normalize --> TechInd[Calculate Technical Indicators<br/>MACD, RSI, BB, ATR, EMA]
    Sentiment --> Smooth[Sentiment Smoothing<br/>Exponential MA]
    
    TechInd --> Window[Create Sequence Windows<br/>60 time steps]
    Smooth --> Window
    
    Window --> Model[Probabilistic DL Model<br/>LSTM + Transformer]
    
    Model --> Predict1[Direction Prediction<br/>Up/Down Probability]
    Model --> Predict2[Level-Reach Probability<br/>Today's Open, Yesterday's Open/Close]
    Model --> Confidence[Confidence Intervals<br/>Uncertainty Quantification]
    
    Predict1 --> API[FastAPI Endpoint<br/>/predict]
    Predict2 --> API
    Confidence --> API
    
    API --> TradingView[TradingView<br/>Pine Script]
    API --> Dashboard[Dash Dashboard<br/>Visualization]
    
    TradingView --> Trade[Execute Trade<br/>If Confidence > 70%]
    Dashboard --> Monitor[Monitor Performance]
    
    Trade --> Log[Log Results]
    Monitor --> Log
    Log --> Evaluate[Evaluate Performance<br/>Daily Reports]
    
    Evaluate --> Retrain{Model Drift?<br/>Deviation > 5%}
    Retrain -->|Yes| RetrainModel[Retrain Model<br/>With Latest Data]
    Retrain -->|No| Collect
    RetrainModel --> Model
```

## 4. Deployment Architecture

```mermaid
graph TB
    subgraph "Client Layer"
        C1[TradingView<br/>Pine Script]
        C2[Web Dashboard<br/>Plotly Dash]
        C3[Mobile/Web App<br/>Optional]
    end
    
    subgraph "API Gateway"
        G1[FastAPI/Flask<br/>REST API]
        G2[Authentication<br/>API Keys]
        G3[Rate Limiting]
    end
    
    subgraph "Application Layer"
        A1[Prediction Service<br/>Model Inference]
        A2[Feature Engineering<br/>Real-time Processing]
        A3[Data Synchronization<br/>Time Alignment]
    end
    
    subgraph "Model Layer"
        M1[Trained Model<br/>LSTM + Transformer]
        M2[Model Cache<br/>In-memory]
        M3[Monte Carlo<br/>Dropout Sampling]
    end
    
    subgraph "Data Layer"
        D1[Alpaca API<br/>Live Data Stream]
        D2[Data Cache<br/>Redis/Memory]
        D3[Historical Data<br/>Parquet Files]
        D4[Sentiment Cache<br/>News Data]
    end
    
    subgraph "Monitoring Layer"
        Mon1[Performance Metrics<br/>DA, MAE, Latency]
        Mon2[System Health<br/>Uptime, Errors]
        Mon3[Logging<br/>All Predictions]
    end
    
    C1 --> G1
    C2 --> G1
    C3 --> G1
    
    G1 --> G2
    G2 --> G3
    G3 --> A1
    
    A1 --> A2
    A2 --> A3
    A3 --> M1
    
    M1 --> M2
    M2 --> M3
    
    A2 --> D1
    A2 --> D2
    A3 --> D3
    A2 --> D4
    
    A1 --> Mon1
    G1 --> Mon2
    M3 --> Mon3
```

## 5. Component Interaction Sequence

```mermaid
sequenceDiagram
    participant TV as TradingView
    participant API as FastAPI Server
    participant FE as Feature Engine
    participant Model as DL Model
    participant Alpaca as Alpaca API
    participant News as News API
    participant DB as Data Storage
    
    TV->>API: Request Prediction (symbol, timestamp)
    API->>FE: Get Features for Symbol
    
    FE->>Alpaca: Fetch Latest OHLCV Data
    Alpaca-->>FE: Return Minute-level Data
    
    FE->>News: Fetch Recent Headlines
    News-->>FE: Return News Data
    
    FE->>FE: Calculate Technical Indicators
    FE->>FE: Process Sentiment (FinBERT)
    FE->>FE: Create Sequence Window (60 steps)
    
    FE->>DB: Check Cache for Historical Data
    DB-->>FE: Return Cached Features
    
    FE-->>API: Return Feature Vector
    
    API->>Model: Predict (feature_vector)
    
    Model->>Model: LSTM Processing
    Model->>Model: Transformer Attention
    Model->>Model: Feature Fusion
    Model->>Model: Monte Carlo Dropout (100 samples)
    
    Model-->>API: Return Predictions:<br/>- Direction Probability<br/>- Level-Reach Probability<br/>- Confidence Intervals
    
    API->>API: Format Response
    API-->>TV: Return JSON Response
    
    TV->>TV: Display on Chart
    TV->>TV: Trigger Alert (if confidence > 70%)
```

## 6. Training Pipeline Architecture

```mermaid
graph TD
    subgraph "Data Preparation"
        DP1[Historical Data<br/>3+ months]
        DP2[Preprocessing<br/>Clean, Normalize]
        DP3[Feature Engineering<br/>Indicators + Sentiment]
        DP4[Sequence Creation<br/>60-step windows]
        DP5[Train/Val/Test Split<br/>70/20/10]
    end
    
    subgraph "Model Training"
        MT1[Initialize Model<br/>LSTM + Transformer]
        MT2[Adam Optimizer<br/>LR: 0.001]
        MT3[Combined Loss<br/>MSE + NLL]
        MT4[Early Stopping<br/>15 epochs patience]
        MT5[Batch Training<br/>Sequence windows]
    end
    
    subgraph "Validation"
        V1[Validation Set<br/>Performance Check]
        V2[Metrics Calculation<br/>DA, MAE, RMSE, Brier]
        V3[Model Checkpoint<br/>Save Best Model]
    end
    
    subgraph "Evaluation"
        E1[Test Set<br/>Unseen Data]
        E2[Backtesting<br/>Walk-forward]
        E3[Baseline Comparison<br/>ARIMA, RF, LSTM]
        E4[Performance Report]
    end
    
    DP1 --> DP2
    DP2 --> DP3
    DP3 --> DP4
    DP4 --> DP5
    
    DP5 --> MT1
    MT1 --> MT2
    MT2 --> MT3
    MT3 --> MT4
    MT4 --> MT5
    
    MT5 --> V1
    V1 --> V2
    V2 --> V3
    V3 --> MT5
    
    V3 --> E1
    E1 --> E2
    E2 --> E3
    E3 --> E4
```

## 7. System Component Diagram

```mermaid
graph TB
    subgraph "External Services"
        EXT1[Alpaca API]
        EXT2[Yahoo Finance]
        EXT3[News APIs]
        EXT4[TradingView Platform]
    end
    
    subgraph "Core System Components"
        COMP1[Data Collector<br/>Module]
        COMP2[Data Preprocessor<br/>Module]
        COMP3[Feature Engineer<br/>Module]
        COMP4[Model Trainer<br/>Module]
        COMP5[Model Inference<br/>Engine]
        COMP6[API Server<br/>FastAPI/Flask]
        COMP7[Dashboard<br/>Plotly Dash]
    end
    
    subgraph "Storage Components"
        STOR1[Raw Data<br/>Storage]
        STOR2[Processed Data<br/>Storage]
        STOR3[Model Weights<br/>Storage]
        STOR4[Prediction Logs<br/>Storage]
        STOR5[Cache<br/>Redis/Memory]
    end
    
    subgraph "Monitoring Components"
        MON1[Metrics Collector]
        MON2[Performance Analyzer]
        MON3[Alert System]
        MON4[Logging Service]
    end
    
    EXT1 --> COMP1
    EXT2 --> COMP1
    EXT3 --> COMP1
    
    COMP1 --> STOR1
    STOR1 --> COMP2
    COMP2 --> STOR2
    STOR2 --> COMP3
    COMP3 --> COMP4
    COMP4 --> STOR3
    
    STOR3 --> COMP5
    COMP3 --> COMP5
    COMP5 --> COMP6
    COMP6 --> EXT4
    COMP6 --> COMP7
    
    COMP5 --> STOR5
    COMP6 --> MON1
    MON1 --> MON2
    MON2 --> MON3
    COMP6 --> MON4
    MON4 --> STOR4
```

## 8. Real-time Processing Flow

```mermaid
flowchart LR
    subgraph "Input Stream"
        I1[Live Market Data<br/>1-minute bars]
        I2[News Feed<br/>Real-time headlines]
    end
    
    subgraph "Processing Pipeline"
        P1[Data Validation<br/>±3σ filter]
        P2[Feature Calculation<br/>Technical Indicators]
        P3[Sentiment Processing<br/>FinBERT]
        P4[Window Creation<br/>60-step sequence]
        P5[Model Inference<br/>LSTM + Transformer]
        P6[Monte Carlo Sampling<br/>100 forward passes]
    end
    
    subgraph "Output Stream"
        O1[Direction Probability<br/>Up/Down]
        O2[Level-Reach Probability<br/>Key price levels]
        O3[Confidence Intervals<br/>95% CI]
    end
    
    subgraph "Action Layer"
        ACT1[Signal Generation<br/>If confidence > 70%]
        ACT2[Alert Trigger<br/>TradingView]
        ACT3[Log Prediction<br/>Database]
    end
    
    I1 --> P1
    I2 --> P3
    P1 --> P2
    P2 --> P4
    P3 --> P4
    P4 --> P5
    P5 --> P6
    P6 --> O1
    P6 --> O2
    P6 --> O3
    O1 --> ACT1
    O2 --> ACT1
    O3 --> ACT1
    ACT1 --> ACT2
    ACT1 --> ACT3
```

## Architecture Notes

### Key Design Principles:
1. **Modularity:** Each component is independent and can be developed/tested separately
2. **Scalability:** System can handle multiple assets and concurrent requests
3. **Reliability:** Multiple data sources and caching for fault tolerance
4. **Low Latency:** Caching, optimized inference, and efficient data pipelines
5. **Observability:** Comprehensive monitoring and logging at every layer

### Technology Stack:
- **Backend:** Python 3.11, FastAPI/Flask
- **ML Framework:** TensorFlow/PyTorch
- **Data Processing:** Pandas, NumPy
- **Technical Indicators:** pandas_ta
- **Visualization:** Plotly Dash, Matplotlib
- **Storage:** Parquet files, Redis (cache)
- **APIs:** Alpaca, Yahoo Finance, News APIs
- **Deployment:** Docker (optional), Cloud/Edge

### Performance Targets:
- **Inference Latency:** <200 ms end-to-end
- **System Uptime:** ≥99%
- **Prediction Accuracy:** ≥90% directional accuracy
- **Throughput:** Handle 100+ requests/minute

