# -*- coding: utf-8 -*-
import pandas as pd
from flask import Flask, render_template, send_from_directory, jsonify, request
from flask_cors import CORS
import numpy as np
import os
import logging
import json
from datetime import datetime, timedelta

# --- Configuration ---
app = Flask(__name__)
CORS(app)
DATA_DIR = 'data'
LIVE_DATA_PATH = os.path.join(DATA_DIR, 'live_data.csv')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Predicted Demand = 33.7221 + 0.7439(Units Sold) + 0.0516(Units Ordered) + 8.6003(Promotion) − 14.5887(Epidemic)
COEFFICIENTS = {
    'Intercept': 33.7221,
    'Units Sold': 0.7439,
    'Units Ordered': 0.0516,
    'Promotion': 8.6003,
    'Epidemic': -14.5887
}

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if pd.isna(obj):
            return None
        return super(NpEncoder, self).default(obj)

def normalize_column_names(df):
    """
    Normalizes column names to handle variations like:
    - 'Units Sold' vs 'UnitsSold' vs 'units_sold' vs 'Sum of Units Sold'
    - 'Product_ID' vs 'ProductID' vs 'Product ID'
    """
    column_mapping = {}
    
    # Define mapping patterns (target_name: list of possible variations)
    patterns = {
        'Units Sold': ['units sold', 'unitssold', 'units_sold', 'sum of units sold', 'sold units', 'qty sold', 'quantity sold'],
        'Units Ordered': ['units ordered', 'unitsordered', 'units_ordered', 'ordered units', 'qty ordered', 'quantity ordered', 'order quantity'],
        'Product_ID': ['product_id', 'productid', 'product id', 'product', 'prod_id', 'sku'],
        'Region': ['region', 'area', 'territory', 'location'],
        'Category': ['category', 'product category', 'cat', 'type'],
        'Price': ['price', 'unit price', 'unitprice', 'selling price'],
        'Revenue': ['revenue', 'sales', 'total revenue', 'total sales', 'sales amount'],
        'Promotion': ['promotion', 'promo', 'is_promotion', 'promotional'],
        'Epidemic': ['epidemic', 'crisis', 'emergency', 'outbreak'],
        'Demand': ['demand', 'forecasted demand', 'predicted demand', 'forecast'],
        'Date': ['date', 'transaction date', 'sale date', 'order date']
    }
    
    actual_columns = {col.lower().strip(): col for col in df.columns}
    
    for target_name, variations in patterns.items():
        for variation in variations:
            if variation in actual_columns:
                original_col = actual_columns[variation]
                if original_col != target_name:  
                    column_mapping[original_col] = target_name
                break
    
    if column_mapping:
        logger.info(f"Column mapping applied: {column_mapping}")
        df = df.rename(columns=column_mapping)
    
    logger.info(f"Final columns after normalization: {df.columns.tolist()}")
    return df

# --- Initialization & Dummy Data Creation ---
def create_initial_data():
    """Creates dummy data if the live data file is missing or empty."""
    if not os.path.exists(LIVE_DATA_PATH) or os.path.getsize(LIVE_DATA_PATH) == 0:
        logger.info(f"Creating initial dummy data at: {LIVE_DATA_PATH}")
        np.random.seed(42)
        dates = pd.date_range(start='2022-01-01', end='2024-12-31', freq='D')
        n_records = len(dates)
        
        products = [f'P{i:04d}' for i in range(1, 21)]
        regions = ['North', 'South', 'East', 'West']
        categories = ['Groceries', 'Clothing', 'Furniture', 'Toys', 'Electronics']
        
        data = {
            'Date': np.random.choice(dates, n_records),
            'Product_ID': np.random.choice(products, n_records),
            'Region': np.random.choice(regions, n_records),
            'Category': np.random.choice(categories, n_records),
            'Units Sold': np.random.randint(50, 500, n_records),
            'Units Ordered': np.random.randint(100, 600, n_records),
            'Price': np.random.uniform(50, 200, n_records).round(2),
            'Revenue': np.random.randint(5000, 50000, n_records),
            'Promotion': np.random.choice([0, 1], n_records, p=[0.7, 0.3]),
            'Epidemic': np.random.choice([0, 1], n_records, p=[0.95, 0.05])
        }
        
        df = pd.DataFrame(data)
        df['Demand'] = df['Units Sold'] + np.random.randint(10, 50, n_records)
        df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
        df.to_csv(LIVE_DATA_PATH, index=False)
        return True
    return False

def safe_numeric_column(df, col_name):
    """Converts column to numeric, handling errors gracefully."""
    if col_name not in df.columns:
        return None
    try:
        return pd.to_numeric(df[col_name], errors='coerce')
    except:
        return None

# --- ANALYSIS FUNCTIONS ---

def calculate_data_health(df):
    """Data Health Check: Basic statistics about data quality."""
    try:
        if df.empty:
            return {
                "total_rows": "0",
                "total_columns": "0",
                "missing_summary": "No data"
            }
        
        total_rows = len(df)
        total_cols = len(df.columns)
        
        # Calculate missing values
        missing_counts = df.isnull().sum()
        total_missing = missing_counts.sum()
        
        if total_missing == 0:
            missing_summary = "No missing values ✓"
        else:
            missing_pct = (total_missing / (total_rows * total_cols)) * 100
            missing_summary = f"{total_missing:,} cells ({missing_pct:.2f}%)"
        
        return {
            "total_rows": f"{total_rows:,}",
            "total_columns": str(total_cols),
            "missing_summary": missing_summary
        }
    except Exception as e:
        logger.error(f"Error in calculate_data_health: {e}", exc_info=True)
        return {
            "total_rows": "Error",
            "total_columns": "Error",
            "missing_summary": f"Error: {str(e)}"
        }

def calculate_promotion_roi(df):
    """F4: Analyzes ROI of promotions by comparing revenue during promo vs non-promo periods."""
    try:
        if df.empty:
            return {
                "avg_revenue_with_promo": "N/A",
                "avg_revenue_without_promo": "N/A",
                "roi_percentage": "N/A"
            }
        
        if 'Promotion' not in df.columns or 'Revenue' not in df.columns:
            return {
                "avg_revenue_with_promo": "Missing columns",
                "avg_revenue_without_promo": "Missing columns",
                "roi_percentage": "N/A"
            }
        
        df_copy = df.copy()
        
        df_copy['Promotion'] = safe_numeric_column(df_copy, 'Promotion')
        df_copy['Revenue'] = safe_numeric_column(df_copy, 'Revenue')
        
        df_copy = df_copy.dropna(subset=['Promotion', 'Revenue'])
        
        if df_copy.empty:
            return {
                "avg_revenue_with_promo": "No valid data",
                "avg_revenue_without_promo": "No valid data",
                "roi_percentage": "N/A"
            }
        
        promo_data = df_copy[df_copy['Promotion'] == 1]['Revenue']
        non_promo_data = df_copy[df_copy['Promotion'] == 0]['Revenue']
        
        if promo_data.empty or non_promo_data.empty:
            return {
                "avg_revenue_with_promo": "No promo data" if promo_data.empty else f"฿{promo_data.mean():,.2f}",
                "avg_revenue_without_promo": "No non-promo data" if non_promo_data.empty else f"฿{non_promo_data.mean():,.2f}",
                "roi_percentage": "Insufficient data"
            }
        
        promo_revenue = promo_data.mean()
        non_promo_revenue = non_promo_data.mean()
        
        if non_promo_revenue > 0:
            roi = ((promo_revenue - non_promo_revenue) / non_promo_revenue) * 100
        else:
            roi = 0
        
        return {
            "avg_revenue_with_promo": f"฿{promo_revenue:,.2f}",
            "avg_revenue_without_promo": f"฿{non_promo_revenue:,.2f}",
            "roi_percentage": f"{roi:+.2f}%"
        }
    except Exception as e:
        logger.error(f"Error in calculate_promotion_roi: {e}", exc_info=True)
        return {
            "avg_revenue_with_promo": "Error",
            "avg_revenue_without_promo": "Error",
            "roi_percentage": f"Error: {str(e)}"
        }

def calculate_demand_forecast_accuracy(df):
    """F5: Measures how accurately we're predicting demand (Demand vs Units Sold)."""
    try:
        if df.empty:
            return {"forecast_accuracy": "N/A", "avg_error": "N/A"}
        
        if 'Demand' not in df.columns or 'Units Sold' not in df.columns:
            return {"forecast_accuracy": "Missing columns", "avg_error": "N/A"}
        
        df_copy = df.copy()
        
        df_copy['Demand'] = safe_numeric_column(df_copy, 'Demand')
        df_copy['Units Sold'] = safe_numeric_column(df_copy, 'Units Sold')
        
        df_copy = df_copy.dropna(subset=['Demand', 'Units Sold'])
        
        if df_copy.empty:
            return {"forecast_accuracy": "No valid data", "avg_error": "N/A"}
        
        df_copy['Forecast_Error'] = abs(df_copy['Demand'] - df_copy['Units Sold'])
        df_copy['Accuracy'] = 1 - (df_copy['Forecast_Error'] / df_copy['Demand'].replace(0, np.nan))
        
        df_copy = df_copy.replace([np.inf, -np.inf], np.nan).dropna(subset=['Accuracy'])
        
        if df_copy.empty:
            return {"forecast_accuracy": "Cannot calculate", "avg_error": "N/A"}
        
        avg_accuracy = df_copy['Accuracy'].mean() * 100
        avg_error = df_copy['Forecast_Error'].mean()
        
        return {
            "forecast_accuracy": f"{avg_accuracy:.2f}%",
            "avg_error": f"{avg_error:.0f} units"
        }
    except Exception as e:
        logger.error(f"Error in calculate_demand_forecast_accuracy: {e}", exc_info=True)
        return {"forecast_accuracy": f"Error: {str(e)}", "avg_error": "N/A"}

def calculate_regional_performance(df):
    """F6: Ranks regions by revenue per unit sold (efficiency)."""
    try:
        if df.empty:
            return {"top_region": "N/A", "revenue_per_unit": "N/A"}
        
        if 'Region' not in df.columns or 'Revenue' not in df.columns or 'Units Sold' not in df.columns:
            return {"top_region": "Missing columns", "revenue_per_unit": "N/A"}
        
        df_copy = df.copy()
        
        df_copy['Revenue'] = safe_numeric_column(df_copy, 'Revenue')
        df_copy['Units Sold'] = safe_numeric_column(df_copy, 'Units Sold')
        
        df_copy = df_copy.dropna(subset=['Region', 'Revenue', 'Units Sold'])
        
        if df_copy.empty:
            return {"top_region": "No valid data", "revenue_per_unit": "N/A"}
        
        regional_stats = df_copy.groupby('Region').agg({
            'Revenue': 'sum',
            'Units Sold': 'sum'
        })
        
        regional_stats['Revenue_Per_Unit'] = regional_stats['Revenue'] / regional_stats['Units Sold'].replace(0, np.nan)
        regional_stats = regional_stats.dropna(subset=['Revenue_Per_Unit'])
        
        if regional_stats.empty:
            return {"top_region": "Cannot calculate", "revenue_per_unit": "N/A"}
        
        top_region = regional_stats['Revenue_Per_Unit'].idxmax()
        top_value = regional_stats['Revenue_Per_Unit'].max()
        
        return {
            "top_region": str(top_region),
            "revenue_per_unit": f"฿{top_value:,.2f}"
        }
    except Exception as e:
        logger.error(f"Error in calculate_regional_performance: {e}", exc_info=True)
        return {"top_region": f"Error: {str(e)}", "revenue_per_unit": "N/A"}

def calculate_epidemic_impact_analysis(df):
    """F7: Analyzes how epidemic affects units sold and revenue."""
    try:
        if df.empty:
            return {
                "units_sold_impact": "N/A",
                "revenue_impact": "N/A"
            }
        
        if 'Epidemic' not in df.columns:
            return {
                "units_sold_impact": "Missing Epidemic column",
                "revenue_impact": "Missing Epidemic column"
            }
        
        df_copy = df.copy()
        
        df_copy['Epidemic'] = safe_numeric_column(df_copy, 'Epidemic')
        
        if 'Units Sold' in df_copy.columns:
            df_copy['Units Sold'] = safe_numeric_column(df_copy, 'Units Sold')
        if 'Revenue' in df_copy.columns:
            df_copy['Revenue'] = safe_numeric_column(df_copy, 'Revenue')
        
        df_copy = df_copy.dropna(subset=['Epidemic'])
        
        if df_copy.empty:
            return {
                "units_sold_impact": "No valid data",
                "revenue_impact": "No valid data"
            }
        
        # Compare units sold during epidemic vs normal
        units_impact = "N/A"
        if 'Units Sold' in df_copy.columns:
            epidemic_units = df_copy[df_copy['Epidemic'] == 1]['Units Sold'].dropna()
            normal_units = df_copy[df_copy['Epidemic'] == 0]['Units Sold'].dropna()
            
            if not epidemic_units.empty and not normal_units.empty:
                units_epidemic = epidemic_units.mean()
                units_normal = normal_units.mean()
                if units_normal > 0:
                    units_impact = f"{((units_epidemic - units_normal) / units_normal * 100):+.2f}%"
        
        revenue_impact = "N/A"
        if 'Revenue' in df_copy.columns:
            epidemic_revenue = df_copy[df_copy['Epidemic'] == 1]['Revenue'].dropna()
            normal_revenue = df_copy[df_copy['Epidemic'] == 0]['Revenue'].dropna()
            
            if not epidemic_revenue.empty and not normal_revenue.empty:
                revenue_epidemic = epidemic_revenue.mean()
                revenue_normal = normal_revenue.mean()
                if revenue_normal > 0:
                    revenue_impact = f"{((revenue_epidemic - revenue_normal) / revenue_normal * 100):+.2f}%"
        
        return {
            "units_sold_impact": units_impact,
            "revenue_impact": revenue_impact
        }
    except Exception as e:
        logger.error(f"Error in calculate_epidemic_impact_analysis: {e}", exc_info=True)
        return {
            "units_sold_impact": f"Error: {str(e)}",
            "revenue_impact": "Error"
        }

def calculate_pricing_effectiveness(df):
    """F8: Analyzes price sensitivity - correlation between price changes and units sold."""
    try:
        if df.empty:
            return {"price_correlation": "N/A", "optimal_price_range": "N/A"}
        
        if 'Price' not in df.columns or 'Units Sold' not in df.columns:
            return {"price_correlation": "Missing columns", "optimal_price_range": "N/A"}
        
        df_copy = df.copy()

        df_copy['Price'] = safe_numeric_column(df_copy, 'Price')
        df_copy['Units Sold'] = safe_numeric_column(df_copy, 'Units Sold')
        
        df_copy = df_copy.dropna(subset=['Price', 'Units Sold'])
        
        if df_copy.empty or len(df_copy) < 2:
            return {"price_correlation": "Insufficient data", "optimal_price_range": "N/A"}
        
        correlation = df_copy['Price'].corr(df_copy['Units Sold'])
        
        # Find price range with highest units sold
        try:
            df_copy['Price_Bin'] = pd.cut(df_copy['Price'], bins=5)
            price_analysis = df_copy.groupby('Price_Bin', observed=True)['Units Sold'].mean()
            
            if not price_analysis.empty:
                optimal_bin = price_analysis.idxmax()
                optimal_range = str(optimal_bin)
            else:
                optimal_range = "Cannot determine"
        except:
            optimal_range = "Cannot determine"
        
        return {
            "price_correlation": f"{correlation:.3f}" if not pd.isna(correlation) else "N/A",
            "optimal_price_range": optimal_range
        }
    except Exception as e:
        logger.error(f"Error in calculate_pricing_effectiveness: {e}", exc_info=True)
        return {"price_correlation": f"Error: {str(e)}", "optimal_price_range": "N/A"}

def calculate_category_trends(df):
    """F9: Identifies trending categories (growing vs declining)."""
    try:
        if df.empty:
            return {"trending_up": "N/A", "trending_down": "N/A"}
        
        if 'Category' not in df.columns or 'Date' not in df.columns or 'Units Sold' not in df.columns:
            return {"trending_up": "Missing columns", "trending_down": "N/A"}
        
        df_copy = df.copy()
        
        # Convert to datetime
        df_copy['Date'] = pd.to_datetime(df_copy['Date'], errors='coerce')
        df_copy['Units Sold'] = safe_numeric_column(df_copy, 'Units Sold')
        
        df_copy = df_copy.dropna(subset=['Date', 'Category', 'Units Sold'])
        
        if df_copy.empty:
            return {"trending_up": "No valid data", "trending_down": "N/A"}
        
        df_copy['Month'] = df_copy['Date'].dt.to_period('M')
        
        # Compare last 3 months vs previous 3 months
        latest_month = df_copy['Month'].max()
        
        if pd.isna(latest_month):
            return {"trending_up": "Cannot determine", "trending_down": "N/A"}
        
        last_3_months = df_copy[df_copy['Month'] >= (latest_month - 2)]
        prev_3_months = df_copy[(df_copy['Month'] < (latest_month - 2)) & (df_copy['Month'] >= (latest_month - 5))]
        
        if last_3_months.empty or prev_3_months.empty:
            return {"trending_up": "Insufficient date range", "trending_down": "N/A"}
        
        current_sales = last_3_months.groupby('Category')['Units Sold'].sum()
        previous_sales = prev_3_months.groupby('Category')['Units Sold'].sum()
        
        # Find common categories
        common_cats = current_sales.index.intersection(previous_sales.index)
        
        if len(common_cats) == 0:
            return {"trending_up": "No common categories", "trending_down": "N/A"}
        
        growth = {}
        for cat in common_cats:
            if previous_sales[cat] > 0:
                growth[cat] = ((current_sales[cat] - previous_sales[cat]) / previous_sales[cat] * 100)
        
        if not growth:
            return {"trending_up": "Cannot calculate growth", "trending_down": "N/A"}
        
        growth_series = pd.Series(growth).sort_values(ascending=False)
        
        trending_up = growth_series.head(2)
        trending_down = growth_series.tail(2)
        
        up_text = [f"{cat} (+{val:.1f}%)" for cat, val in trending_up.items() if val > 0]
        down_text = [f"{cat} ({val:.1f}%)" for cat, val in trending_down.items() if val < 0]
        
        return {
            "trending_up": ", ".join(up_text) if up_text else "No significant growth",
            "trending_down": ", ".join(down_text) if down_text else "No significant decline"
        }
    except Exception as e:
        logger.error(f"Error in calculate_category_trends: {e}", exc_info=True)
        return {"trending_up": f"Error: {str(e)}", "trending_down": "Error"}

def run_analysis(df):
    """Runs all NEW analysis functions."""
    if df.empty:
        logger.warning("Empty DataFrame. Returning default metrics.")
        return {
            "promotion_roi": {"avg_revenue_with_promo": "N/A", "avg_revenue_without_promo": "N/A", "roi_percentage": "N/A"},
            "forecast_accuracy": {"forecast_accuracy": "N/A", "avg_error": "N/A"},
            "regional_performance": {"top_region": "N/A", "revenue_per_unit": "N/A"},
            "epidemic_impact": {"units_sold_impact": "N/A", "revenue_impact": "N/A"},
            "pricing_effectiveness": {"price_correlation": "N/A", "optimal_price_range": "N/A"},
            "category_trends": {"trending_up": "N/A", "trending_down": "N/A"}
        }
    
    logger.info(f"Running analysis on {len(df)} rows with columns: {df.columns.tolist()}")
    
    return {
        "promotion_roi": calculate_promotion_roi(df),
        "forecast_accuracy": calculate_demand_forecast_accuracy(df),
        "regional_performance": calculate_regional_performance(df),
        "epidemic_impact": calculate_epidemic_impact_analysis(df),
        "pricing_effectiveness": calculate_pricing_effectiveness(df),
        "category_trends": calculate_category_trends(df)
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
                df = normalize_column_names(df)
                logger.info(f"Loaded {len(df)} rows from existing file")
                logger.info(f"Columns: {df.columns.tolist()}")
                analysis_summary = run_analysis(df)
                data_health = calculate_data_health(df)
                response_data = {
                    "message": "Loaded existing data and ran analysis",
                    "analysis_summary": analysis_summary,
                    "data_health": data_health
                }
                return app.response_class(
                    response=json.dumps(response_data, cls=NpEncoder),
                    status=200,
                    mimetype='application/json'
                )
            except Exception as e:
                logger.error(f"Error reading existing file: {e}", exc_info=True)
                create_initial_data()
                df = pd.read_csv(LIVE_DATA_PATH)
                df = normalize_column_names(df)
                analysis_summary = run_analysis(df)
                data_health = calculate_data_health(df)
                response_data = {
                    "message": f"Created new dummy data",
                    "analysis_summary": analysis_summary,
                    "data_health": data_health
                }
                return app.response_class(
                    response=json.dumps(response_data, cls=NpEncoder),
                    status=200,
                    mimetype='application/json'
                )
        
        create_initial_data()
        df = pd.read_csv(LIVE_DATA_PATH)
        df = normalize_column_names(df)
        analysis_summary = run_analysis(df)
        data_health = calculate_data_health(df)
        response_data = {
            "message": "Created initial dummy data",
            "analysis_summary": analysis_summary,
            "data_health": data_health
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
        
        logger.info(f"Uploaded file has {len(df)} rows and columns: {df.columns.tolist()}")
        
        # Normalize column names
        df = normalize_column_names(df)
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.normalize()

        df.to_csv(LIVE_DATA_PATH, index=False)
        logger.info(f"File '{uploaded_file.filename}' successfully saved with normalized columns")
        
        analysis_summary = run_analysis(df)
        data_health = calculate_data_health(df)
        
        response_data = {
            "message": f"File '{uploaded_file.filename}' successfully processed",
            "analysis_summary": analysis_summary,
            "data_health": data_health
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
    """NEW FORMULA: Predicted Demand = 33.7221 + 0.7439(Units Sold) + 0.0516(Units Ordered) + 8.6003(Promotion) − 14.5887(Epidemic)"""
    logger.info("Received request to /api/predict")
    data = request.json
    
    try:
        units_sold = float(data.get('units_sold', 0))
        units_ordered = float(data.get('units_ordered', 0))
        promotion = float(data.get('promotion', 0))
        epidemic = float(data.get('epidemic', 0))
        
        if any(v is None for v in [units_sold, units_ordered, promotion, epidemic]):
            return jsonify({"error": "Missing or invalid input parameters"}), 400

        prediction = (
            COEFFICIENTS['Intercept'] +
            (COEFFICIENTS['Units Sold'] * units_sold) +
            (COEFFICIENTS['Units Ordered'] * units_ordered) +
            (COEFFICIENTS['Promotion'] * promotion) +
            (COEFFICIENTS['Epidemic'] * epidemic)
        )
        
        explanation = f"Based on: Sold={units_sold:.0f}, Ordered={units_ordered:.0f}, Promo={'Yes' if promotion == 1 else 'No'}, Epidemic={'Yes' if epidemic == 1 else 'No'}"

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
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # ใช้ PORT ที่ Render กำหนด ถ้าไม่มีใช้ 5000
    app.run(host="0.0.0.0", port=port, debug=False)
