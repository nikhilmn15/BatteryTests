import pandas as pd
import numpy as np
from pathlib import Path

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans physical battery log records by removing zero-valued instrumentation artifacts 
    and safely filling missing values in charging protocol metadata parameters.
    """
    df_clean = df.copy()
    
    # 1. Clean column spacing anomalies
    df_clean.columns = df_clean.columns.str.strip()
    
    # 2. Fix the Cycle 1 Instrumentation Dead-Zone
    # Cycle 1 often registers absolute 0.0 values across resistance, temperature, and capacity 
    # due to hardware sensor connection/handshake lags. We filter these out.
    df_clean = df_clean[df_clean['cycle'] > 1]
    
    # 3. Handle specific missing charging protocol configuration indices
    protocol_cols = ['C1', 'Q1', 'C2']
    for col in protocol_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median() if df_clean[col].notnull().any() else 0.0)
            
    # 4. Handle rare missing target tracking parameters
    if 'cycle_life' in df_clean.columns:
        df_clean['cycle_life'] = df_clean['cycle_life'].fillna(df_clean.groupby('battery_id')['cycle'].transform('max'))
        
    return df_clean


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers physical, thermodynamic, and electrochemical degradation features 
    from raw cycle lifetime logs and builds the Approach B Target matrix.
    """
    df_feats = df.copy()
    df_feats.columns = df_feats.columns.str.strip()
    
    # ------------------------------------------------------------------------
    # STEP 1: APPROACH B NON-LINEAR TARGET ENGINEERING
    # ------------------------------------------------------------------------
    # Establish true baseline fresh capacity by fetching the max discharge capacity recorded per pack
    nominal_capacities = df_feats.groupby('battery_id')['QD'].max().reset_index().rename(columns={'QD': 'Nominal_QD_Cap'})
    df_feats = df_feats.merge(nominal_capacities, on='battery_id')
    
    # Dynamic Capacity State of Health (SOH) Ratio
    df_feats['target_soh'] = df_feats['QD'] / df_feats['Nominal_QD_Cap']
    
    # Extract historical velocity: Capacity degradation rate per elapsed cycle
    capacity_lost_fraction = (1.0 - df_feats['target_soh']).clip(lower=1e-5)
    cycles_elapsed = df_feats['cycle'].clip(lower=1)
    degradation_rate_per_cycle = capacity_lost_fraction / cycles_elapsed
    
    # Calculate capacity runway remaining until the standard 80% EOL cliff boundary
    soh_runway_left = (df_feats['target_soh'] - 0.80).clip(lower=0)
    
    # Approach B RUL Calculation
    # Bounded with an upper limit clip of 15,000 cycles to prevent the 
    # "Fresh Battery Singularity" where a near-zero degradation rate forces an explosive RUL.
    df_feats['target_rul_cycles'] = (soh_runway_left / degradation_rate_per_cycle).clip(upper=15000.0)
    
    # Structural Binary Fleet Replacement Flag
    df_feats['target_replace'] = (df_feats['target_soh'] < 0.80).astype(int)
    
    # ------------------------------------------------------------------------
    # STEP 2: PHYSICAL & THERMODYNAMIC FEATURE ENGINEERING
    # ------------------------------------------------------------------------
    # 1. Internal Resistance Growth (Ratio tracking degradation velocity relative to fresh state)
    fresh_ir = df_feats.groupby('battery_id')['IR'].transform('min')
    df_feats['ir_growth_ratio'] = (df_feats['IR'] / (fresh_ir + 1e-6)).clip(upper=5.0)
    
    # 2. Thermal Delta Amplitude (Captures cyclical structural expansion/contraction stress)
    df_feats['thermal_amplitude'] = df_feats['Tmax'] - df_feats['Tmin']
    
    # 3. Thermal Efficiency Index (Capacity extracted relative to average temperature)
    df_feats['thermal_efficiency_index'] = df_feats['QD'] / (df_feats['Tavg'] + 1e-5)
    
    # 4. Coulombic Efficiency Proxy (Ratio of discharge capacity returned to charge capacity invested)
    df_feats['coulombic_efficiency_proxy'] = (df_feats['QD'] / (df_feats['QC'] + 1e-5)).clip(upper=1.0)
    
    # 5. Charging Kinetic Velocity Strain
    df_feats['charge_delivery_velocity'] = df_feats['QC'] / (df_feats['chargetime'] + 1e-5)
    
    return df_feats


if __name__ == "__main__":
    # Absolute path configurations mapped to your local system environment context
    RAW_DIR = Path(r"C:\Users\Nikhil\OneDrive\Desktop\BatteryTests\data\raw")
    PROCESSED_DIR = Path(r"C:\Users\Nikhil\OneDrive\Desktop\BatteryTests\data\processed")
    
    # Guarantee processed destination subdirectory structure exists safely
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    FILES = [
        "Lithium-Ion Battery Cycle Life.csv",
    ]
    
    print(f"🔄 Scanning source environment: {RAW_DIR}")
    
    for file_name in FILES:
        source_path = RAW_DIR / file_name
        
        if source_path.exists():
            print(f"┌── Ingesting File: {source_path.name}")
            raw = pd.read_csv(source_path)
            
            # Execute Pipeline Sequence
            cleaned = clean_data(raw)
            final_features = build_features(cleaned)
            
            # Formulate destination target footprint
            output_name = f"ProcessedV1.csv"
            destination_path = PROCESSED_DIR / output_name
            
            # Save out
            final_features.to_csv(destination_path, index=False)
            print(f"└── Complete. Exported to: {destination_path.resolve()} (Shape: {final_features.shape})")
        else:
            print(f"⚠️ Notice: Missing file target in raw context: {source_path.resolve()}")