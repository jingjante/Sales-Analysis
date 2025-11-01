# -*- coding: utf-8 -*-
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os
import logging 
import json

# --- Configuration ---
app = Flask(__name__)
CORS(app) # Enable CORS for all domains
DATA_DIR = 'data'
LIVE_DATA_PATH = os.path.join(DATA_DIR, 'live_data.csv')

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Define Coefficients from Excel Regression ---
# Demand = 99.68 + (0.1279 * Units Ordered) - (0.0769 * Price) + (18.489 * Promotion) - (38.169 * Epidemic)
COEFFICIENTS = {
    'Intercept': 99.68,
    'Units Ordered': 0.1279,
    'Price': -0.0769, 
    'Promotion': 18.489,
    'Epidemic': -38.169
}
# ----------------------------------------------------

if not os.path.exists(DATA_DIR):
 os.makedirs(DATA_DIR)

# --- JSON Serialization Helper ---
class NpEncoder(json.JSONEncoder):
 def default(self, obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return super(NpEncoder, self).default(obj)

# --- Initialization & Dummy Data Creation ---
def create_initial_data():
    """Checks for or creates dummy data if the live data file is missing or empty."""
    if not os.path.exists(LIVE_DATA_PATH) or os.path.getsize(LIVE_DATA_PATH) == 0:
        logger.info(f"Creating initial dummy data at: {LIVE_DATA_PATH}")
        dates = pd.to_datetime(pd.date_range(end='2025-01-01', periods=76, freq='D'))
        products = [f'P{i:03d}' for i in range(1, 11)]
        
        data = {
            'Date': np.tile(dates, 10)[:760],
            'Product_ID': np.repeat(products, int(760/10))[:760],
            'Price': np.random.randint(50, 200, 760),
            'Units Ordered': np.random.randint(100, 600, 760),
            'Revenue': np.random.randint(50000, 150000, 760),
            'Promotion': np.random.choice([0, 1], 760, p=[0.7, 0.3]),
            'Epidemic': np.random.choice([0, 1], 760, p=[0.9, 0.1])
        }
        df = pd.DataFrame(data)
        # สร้าง Demand ด้วยสูตรเดิมเพื่อให้การวิเคราะห์อื่นๆ ยังคงทำงาน
        df['Demand'] = df['Units Ordered'] + (df['Revenue'] / 1000) * 0.1 + np.random.randint(20, 50, 760)
        df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
        df.loc[df.sample(frac=0.01).index, 'Price'] = np.nan
        df.to_csv(LIVE_DATA_PATH, index=False)
        return True
    return False

# --- Analysis Functions (Modified for Manual Coefficients and Robustness) ---

def calculate_missing_values(df, required_cols):
    """Calculates missing values summary for required columns."""
    if df.empty:
        return [{"column": col, "summary": "0 (0.00%)"} for col in required_cols]
        
    total_rows = len(df)
    missing_summary = []
    for col in required_cols:
        count = 0
        percent = 0.00
        if col in df.columns:
            count = df[col].isnull().sum()
            percent = (count / total_rows) * 100 if total_rows > 0 else 0
        summary_str = f"{count} ({percent:.2f}%)"
        missing_summary.append({"column": col, "summary": summary_str})
    return missing_summary

def get_demand_gap_alert(df):
    """Calculates the top 5 products with the largest positive demand gap."""
    if df.empty or not all(col in df.columns for col in ['Product_ID', 'Demand', 'Units Ordered']):
        return ["N/A (Missing required columns or empty data)"]

    try:
        df_copy = df.copy()
        df_copy['Demand Gap'] = df_copy['Demand'] - df_copy['Units Ordered']
        gap_summary = df_copy.groupby('Product_ID')['Demand Gap'].sum().reset_index()
        gap_summary = gap_summary.sort_values(by='Demand Gap', ascending=False).head(5)
        
        top5_gaps = []
        for index, row in gap_summary.iterrows():
            gap_value = row['Demand Gap'] if pd.notna(row['Demand Gap']) else 0
            if gap_value > 0:
                top5_gaps.append(f"{row['Product_ID']}: {gap_value:,.2f}")
        
        if not top5_gaps:
            return ["No significant gaps found"]
        return top5_gaps
    except Exception as e:
        logger.error(f"Error calculating demand gap: {e}")
        return ["Error during calculation"]


def calculate_elasticity_groups(df):
    """Uses the manual Price Coefficient to determine elasticity group."""
    price_coef = COEFFICIENTS.get('Price')
    if price_coef is None:
        return {"price_coefficient": "N/A", "elasticity_group": "Coefficient Error", "description": "ไม่พบค่าสัมประสิทธิ์ราคา"}
    
    try:
        # ใช้ค่าสัมบูรณ์ของสัมประสิทธิ์เพื่อดูความยืดหยุ่น
        if abs(price_coef) < 1: 
            elasticity_group = "Inelastic"
            description = "การเปลี่ยนแปลงราคาส่งผลต่ออุปสงค์น้อยมาก"
        else:
            elasticity_group = "Elastic"
            description = "การเปลี่ยนแปลงราคาส่งผลต่ออุปสงค์มาก"

        return {
            "price_coefficient": f"{price_coef:.4f}",
            "elasticity_group": elasticity_group,
            "description": description
        }
    except Exception as e:
        logger.error(f"Error calculating elasticity: {e}")
        return {"price_coefficient": "N/A", "elasticity_group": "Error", "description": "เกิดข้อผิดพลาด"}

def calculate_promotion_score(df):
    """Uses the manual Promotion Coefficient to determine promotion effectiveness."""
    score = COEFFICIENTS.get('Promotion', 0)
    
    if score >= 20:
        interpretation = "สูงมาก (20+): โปรโมชั่นมีผลต่อ Demand สูงสุด"
    elif score >= 10:
        interpretation = "ปานกลาง (10-20): โปรโมชั่นมีผลดี"
    elif score >= 0:
        interpretation = "ต่ำ (0-10): โปรโมชั่นอาจไม่มีประสิทธิภาพ"
    else:
        interpretation = "ติดลบ (<0): โปรโมชั่นส่งผลลบ"
        
    return {"score": f"{score:.3f}", "interpretation": interpretation}


def calculate_revenue_volatility(df):
    """Calculates the standard deviation of monthly revenue."""
    if df.empty or 'Date' not in df.columns or 'Revenue' not in df.columns:
        return {"volatility": "N/A", "description": "ต้องมีคอลัมน์ Date และ Revenue หรือข้อมูลว่างเปล่า"}
        
    try:
        df_copy = df.copy()
        df_copy['Date'] = pd.to_datetime(df_copy['Date'], errors='coerce').dt.normalize()
        df_copy = df_copy.dropna(subset=['Date'])
        
        if df_copy.empty:
            return {"volatility": "N/A", "description": "ไม่พบข้อมูลวันที่ที่ถูกต้อง"}

        df_copy['YearMonth'] = df_copy['Date'].dt.to_period('M')
        monthly_revenue = df_copy.groupby('YearMonth')['Revenue'].sum()
        volatility = monthly_revenue.std()
        
        return {"volatility": f"฿{volatility:,.2f}", 
                "description": "ค่าเบี่ยงเบนมาตรฐานของรายได้รายเดือน"}
    except Exception:
        return {"volatility": "N/A", "description": "เกิดข้อผิดพลาดในการคำนวณ"}

def calculate_yoy_growth(df):
    """Calculates year-over-year revenue growth."""
    if df.empty or 'Date' not in df.columns or 'Revenue' not in df.columns:
        return {"yoy_growth": "N/A", "description": "ต้องมีคอลัมน์ Date และ Revenue หรือข้อมูลว่างเปล่า"}

    try:
        df_copy = df.copy()
        df_copy['Date'] = pd.to_datetime(df_copy['Date'], errors='coerce').dt.normalize()
        df_copy = df_copy.dropna(subset=['Date'])
        
        if df_copy.empty:
            return {"yoy_growth": "N/A", "description": "ไม่พบข้อมูลวันที่ที่ถูกต้อง"}
            
        df_copy['Year'] = df_copy['Date'].dt.year
    except Exception:
        return {"yoy_growth": "N/A", "description": "ไม่สามารถแปลงวันที่ได้"}

    valid_years = df_copy['Year'].unique()
    if len(valid_years) < 2:
        return {"yoy_growth": "N/A", "description": "ข้อมูลไม่ครอบคลุม 2 ปี"}

    current_year = df_copy['Year'].max()
    previous_year = current_year - 1
    
    revenue_current = df_copy[df_copy['Year'] == current_year]['Revenue'].sum()
    revenue_previous = df_copy[df_copy['Year'] == previous_year]['Revenue'].sum()

    if revenue_previous == 0:
        yoy_growth = "0.00%"
        description = "รายได้ปีก่อนหน้าเป็นศูนย์"
    else:
        yoy_growth_rate = ((revenue_current - revenue_previous) / revenue_previous) * 100
        description = f"อัตราการเติบโต YoY จากปี {previous_year} ถึง {current_year}"
        yoy_growth = f"{yoy_growth_rate:,.2f}%"

    return {"yoy_growth": yoy_growth, "description": description}

def run_analysis(df):
    """Runs all analysis functions and returns a summary dictionary."""
    if df.empty:
        logger.warning("Attempting to run analysis on empty DataFrame. Returning default empty metrics.")
        empty_cols = ['Date', 'Product_ID', 'Price', 'Units Ordered', 'Revenue', 'Promotion', 'Epidemic', 'Demand']
        return {
            "metrics": {"total_rows": "0", "missing_summary": calculate_missing_values(df, empty_cols), "rolling_avg_revenue": "N/A", "epidemic_revenue": "N/A"},
            "alerts": {"top5_demand_gap": ["No data"]},
            "elasticity": calculate_elasticity_groups(pd.DataFrame()), # Pass empty df, but this function only uses COEFFICIENTS
            "promotion": calculate_promotion_score(pd.DataFrame()),    # Same as above
            "volatility": {"volatility": "N/A", "description": "ต้องมีคอลัมน์ Date และ Revenue หรือข้อมูลว่างเปล่า"},
            "yoy": {"yoy_growth": "N/A", "description": "ข้อมูลไม่ครอบคลุม 2 ปี"}
        }


    required_cols = ['Date', 'Product_ID', 'Price', 'Units Ordered', 'Revenue', 'Promotion', 'Epidemic', 'Demand']
    
    missing_summary = calculate_missing_values(df, [col for col in required_cols if col in df.columns])
    total_rows = len(df)
    
    rolling_avg_revenue = "N/A"
    try:
        df_temp = df.copy()
        df_temp['Date'] = pd.to_datetime(df_temp['Date'], errors='coerce').dt.normalize()
        df_sorted = df_temp.dropna(subset=['Date']).sort_values(by='Date')
        
        if not df_sorted.empty:
            rolling_avg_revenue = df_sorted['Revenue'].rolling(window=30, min_periods=1).mean().iloc[-1]
            rolling_avg_revenue = f"฿{rolling_avg_revenue:,.2f}"
    except:
        pass
    
    top5_gaps = get_demand_gap_alert(df)
    elasticity_data = calculate_elasticity_groups(df)
    promotion_data = calculate_promotion_score(df)
    volatility_data = calculate_revenue_volatility(df)
    yoy_data = calculate_yoy_growth(df)

    epidemic_revenue = "N/A"
    try:
        if 'Epidemic' in df.columns:
            df_temp = df.copy()
            # FIX: Convert Epidemic to numeric (float) to safely handle both 1 and 1.0 from CSV
            df_temp['Epidemic'] = pd.to_numeric(df_temp['Epidemic'], errors='coerce') 
            epidemic_revenue = df_temp[df_temp['Epidemic'] == 1.0]['Revenue'].sum() 
            epidemic_revenue = f"฿{epidemic_revenue:,.2f}"
    except:
        pass

    return {
        "metrics": {
            "total_rows": f"{total_rows:,}",
            "missing_summary": missing_summary,
            "rolling_avg_revenue": rolling_avg_revenue,
            "epidemic_revenue": epidemic_revenue,
        },
        "alerts": {
            "top5_demand_gap": top5_gaps
        },
        "elasticity": elasticity_data,
        "promotion": promotion_data,
        "volatility": volatility_data,
        "yoy": yoy_data
    }

# --- API Routes ---
@app.route('/api/upload', methods=['POST'])
def upload_file():
    logger.info("Received request to /api/upload")
    
    if 'file' not in request.files:
        logger.info("No file uploaded. Loading existing data")
        if os.path.exists(LIVE_DATA_PATH) and os.path.getsize(LIVE_DATA_PATH) > 0:
            try:
                df = pd.read_csv(LIVE_DATA_PATH)
                analysis_summary = run_analysis(df)
                response_data = {
                    "message": "Loaded existing data and ran analysis (Manual Formula)",
                    "analysis_summary": analysis_summary
                }
                return app.response_class(
                    response=json.dumps(response_data, cls=NpEncoder),
                    status=200,
                    mimetype='application/json'
                )
            except Exception as e:
                logger.error(f"Error reading existing file: {e}") 
                # IMPORTANT: If existing file fails, try to create dummy data instead of crashing
                create_initial_data()
                df = pd.read_csv(LIVE_DATA_PATH)
                analysis_summary = run_analysis(df)
                response_data = {
                    "message": f"Error reading old data, created new dummy data: {str(e)}",
                    "analysis_summary": analysis_summary
                }
                return app.response_class(
                    response=json.dumps(response_data, cls=NpEncoder),
                    status=200,
                    mimetype='application/json'
                )

        # No data exists, return analysis on empty DataFrame
        response_data = {
            "message": "Ready to upload. No initial data loaded", 
            "analysis_summary": run_analysis(pd.DataFrame())
        }
        return app.response_class(
            response=json.dumps(response_data, cls=NpEncoder),
            status=200,
            mimetype='application/json'
        )
    
    uploaded_file = request.files['file']
    
    try:
        if uploaded_file.filename.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file, sheet_name=0)
        else:
            return jsonify({"error": "Unsupported file format"}), 400
        
        # Data cleaning and prep
        if 'Units Ordered' not in df.columns or 'Revenue' not in df.columns:
            return jsonify({"error": "Missing critical columns: 'Units Ordered' or 'Revenue'"}), 400

        if 'Demand' not in df.columns:
            df['Demand'] = df['Units Ordered'] + (df['Revenue'] / 1000) * 0.1 + np.random.randint(20, 50, len(df))
            
        if 'Product_ID' not in df.columns:
            df['Product_ID'] = 'P_Def' + (df.index % 5).astype(str)
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.normalize()

        df.to_csv(LIVE_DATA_PATH, index=False)
        logger.info(f"File '{uploaded_file.filename}' successfully saved")
        
        analysis_summary = run_analysis(df)
        
        response_data = {
            "message": f"File '{uploaded_file.filename}' successfully processed (Manual Formula)",
            "analysis_summary": analysis_summary
        }
        return app.response_class(
            response=json.dumps(response_data, cls=NpEncoder),
            status=200,
            mimetype='application/json'
        )

    except Exception as e:
        logger.error(f"Error during file processing: {e}", exc_info=True)
        response_data = {"error": f"Processing error: {str(e)}"}
        return app.response_class(
            response=json.dumps(response_data, cls=NpEncoder),
            status=500,
            mimetype='application/json'
        )

@app.route('/api/predict', methods=['POST'])
def predict_demand():
    logger.info("Received request to /api/predict")
    data = request.json
    
    try:
        units_ordered = data.get('units_ordered')
        price = data.get('price')
        promotion = data.get('promotion')
        epidemic = data.get('epidemic')
        
        # Ensure conversion to float for calculation
        units_ordered = float(units_ordered) if units_ordered is not None else None
        price = float(price) if price is not None else None
        promotion = float(promotion) if promotion is not None else None
        epidemic = float(epidemic) if epidemic is not None else None
        
        if any(v is None for v in [units_ordered, price, promotion, epidemic]):
            return jsonify({"error": "Missing or invalid input parameters"}), 400

        # --- MANUAL CALCULATION based on Excel Formula ---
        # Demand = 99.68 + (0.1279 * Units Ordered) - (0.0769 * Price) + (18.489 * Promotion) - (38.169 * Epidemic)
        prediction = (
            COEFFICIENTS['Intercept'] +
            (COEFFICIENTS['Units Ordered'] * units_ordered) +
            (COEFFICIENTS['Price'] * price) +
            (COEFFICIENTS['Promotion'] * promotion) +
            (COEFFICIENTS['Epidemic'] * epidemic)
        )
        # ----------------------------------------------------
        
        explanation = f"Prediction based on: Units={units_ordered}, Price={price:,.2f}, Promo={'Yes' if promotion == 1 else 'No'}. Epidemic={'Yes' if epidemic == 1 else 'No'}"

        response_data = {
            "predicted_demand": f"{prediction:,.2f}",
            "explanation": explanation
        }
        
        return app.response_class(
            response=json.dumps(response_data, cls=NpEncoder),
            status=200,
            mimetype='application/json'
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 400

# --- Run Server ---
if __name__ == '__main__':
    # MANDATORY: Ensure data directory and initial data exists before running the app
    create_initial_data() 
    
    try:
        # Load data for initial analysis summary to be available
        if os.path.exists(LIVE_DATA_PATH) and os.path.getsize(LIVE_DATA_PATH) > 0:
            df_initial = pd.read_csv(LIVE_DATA_PATH)
            run_analysis(df_initial) 
            logger.info("Initial analysis ran successfully with manual formula")
    except Exception as e:
        logger.warning(f"Could not load initial data or run analysis: {e}")
        
    logger.info(f"Flask server starting. Data path: {LIVE_DATA_PATH}")
    logger.info("Server will run on http://0.0.0.0:5000")
    # NO CHANGE TO THE CONNECTION PART (host='0.0.0.0', port=5000)
    app.run(debug=True, host='0.0.0.0', port=5000)