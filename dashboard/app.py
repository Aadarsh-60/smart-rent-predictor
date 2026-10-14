import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import plotly.express as px
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="Smart Rent Predictor & Advisor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-box {
        background: linear-gradient(135deg, rgba(33, 220, 125, 0.08) 0%, rgba(0, 0, 0, 0.02) 100%);
        border: 1px solid rgba(33, 220, 125, 0.25);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 15px;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #111827;
    }
    .metric-sub {
        font-size: 0.85rem;
        color: #4b5563;
        margin-top: 4px;
    }
    .badge-safe {
        background-color: #d1fae5;
        color: #065f46;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-warning {
        background-color: #fef3c7;
        color: #92400e;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-danger {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Caching for Performance ---
@st.cache_resource
def load_all_artifacts():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    model_path = os.path.join(base_dir, 'models', 'random_forest_model.pkl')
    scaler_path = os.path.join(base_dir, 'models', 'scaler.pkl')
    columns_path = os.path.join(base_dir, 'models', 'training_columns.pkl')
    data_path = os.path.join(base_dir, 'data', 'raw', 'cleaned_data.csv')
    comp_path = os.path.join(base_dir, 'models', 'model_comparison.json')
    feat_path = os.path.join(base_dir, 'models', 'feature_importance.json')
    pred_path = os.path.join(base_dir, 'models', 'test_predictions.json')

    # Fallbacks if running directly from directory
    if not os.path.exists(model_path):
        model_path = os.path.join('models', 'random_forest_model.pkl')
        scaler_path = os.path.join('models', 'scaler.pkl')
        columns_path = os.path.join('models', 'training_columns.pkl')
        data_path = os.path.join('data', 'raw', 'cleaned_data.csv')
        comp_path = os.path.join('models', 'model_comparison.json')
        feat_path = os.path.join('models', 'feature_importance.json')
        pred_path = os.path.join('models', 'test_predictions.json')

    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        training_columns = joblib.load(columns_path)
        df = pd.read_csv(data_path)

        with open(comp_path, 'r') as f:
            comparison = json.load(f)
        with open(feat_path, 'r') as f:
            feature_imp = json.load(f)
        with open(pred_path, 'r') as f:
            predictions_sample = pd.read_json(f)

    except Exception as e:
        st.error(f"Error loading artifacts: {e}. Please run `python src/train_model.py` first.")
        st.stop()

    return model, scaler, training_columns, df, comparison, feature_imp, predictions_sample

model, scaler, TRAINING_COLUMNS, df, MODEL_COMPARISON, FEATURE_IMPORTANCE, TEST_PREDS = load_all_artifacts()

# Calculate helper columns
if 'Price_per_SqFt' not in df.columns:
    df['Price_per_SqFt'] = (df['Rent'] / df['Size']).round(2)

# --- Sidebar Inputs ---
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/real-estate.png", width=64)
    st.title("Smart Rent Predictor")
    st.caption("AI-Powered Real Estate Analytics & Advisory")
    st.markdown("---")

    st.subheader("👤 User Profile")
    salary = st.number_input(
        "Monthly Take-Home Salary (₹)",
        min_value=10000,
        max_value=1500000,
        value=65000,
        step=5000,
        help="Used to calculate the personal finance 30-40% affordability threshold."
    )

    st.markdown("---")
    st.subheader("🏠 Property Preferences")

    city_options = sorted(df['City'].unique())
    selected_city = st.selectbox("Preferred City", options=city_options, index=city_options.index('Mumbai') if 'Mumbai' in city_options else 0)

    col_bhk, col_bath = st.columns(2)
    with col_bhk:
        bhk_options = sorted(df['BHK'].unique())
        selected_bhk = st.selectbox("Rooms (BHK)", options=bhk_options, index=1)
    with col_bath:
        bath_options = sorted(df['Bathroom'].unique())
        selected_bath = st.selectbox("Bathrooms", options=bath_options, index=min(1, len(bath_options)-1))

    # Calculate average size for selected City + BHK as default
    city_bhk_match = df[(df['City'] == selected_city) & (df['BHK'] == selected_bhk)]
    default_size = int(city_bhk_match['Size'].mean()) if not city_bhk_match.empty else 900

    selected_size = st.slider(
        "Super Built-up Area (Sq. Ft.)",
        min_value=200,
        max_value=4000,
        value=default_size,
        step=50,
        help="Estimated square footage of the apartment."
    )

    furnish_options = sorted(df['Furnishing Status'].unique())
    selected_furnishing = st.selectbox("Furnishing Status", options=furnish_options)

    tenant_options = sorted(df['Tenant Preferred'].unique())
    selected_tenant = st.selectbox("Tenant Preference", options=tenant_options)

    st.markdown("---")
    st.caption("Built with ❤️ by Aadarsh | Version 2.0")

# --- App Header ---
st.title("🏡 Smart Rent Predictor & Advisory")
st.markdown("Estimate fair rental prices using Machine Learning, analyze market trends across Indian metros, and evaluate financial affordability.")

# --- Tab Layout ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Rent Estimator & Affordability",
    "📊 Market Analytics & Trends",
    "🤖 Model Benchmarks & Explainability",
    "🧭 Neighborhood Locality Explorer"
])

# ==========================================
# TAB 1: RENT ESTIMATOR & AFFORDABILITY
# ==========================================
with tab1:
    # 1. Prepare Feature Input
    user_input = {
        'BHK': selected_bhk,
        'Size': selected_size,
        'Bathroom': selected_bath,
        'Size_per_BHK': selected_size / selected_bhk,
        'City': selected_city,
        'Furnishing Status': selected_furnishing,
        'Tenant Preferred': selected_tenant
    }
    input_df = pd.DataFrame([user_input])
    input_encoded = pd.get_dummies(input_df, columns=['City', 'Furnishing Status', 'Tenant Preferred'])
    input_reindexed = input_encoded.reindex(columns=TRAINING_COLUMNS, fill_value=0)

    # Scale numeric features
    numeric_cols = ['BHK', 'Size', 'Bathroom', 'Size_per_BHK']
    input_scaled = input_reindexed.copy()
    input_scaled[numeric_cols] = scaler.transform(input_scaled[numeric_cols])

    # Predict with Champion Model
    predicted_rent = float(model.predict(input_scaled)[0])

    # Calculate Prediction Interval from Random Forest individual trees
    tree_preds = np.array([tree.predict(input_scaled.values)[0] for tree in model.estimators_])
    ci_lower = float(np.percentile(tree_preds, 10))
    ci_upper = float(np.percentile(tree_preds, 90))

    # Affordability Calculations
    affordable_rent_30 = salary * 0.30
    affordable_rent_40 = salary * 0.40
    rent_to_income_ratio = (predicted_rent / salary) * 100

    # Key Metrics Cards
    st.subheader("💡 Rent Prediction Summary")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Estimated Rent</div>
            <div class="metric-value">₹{predicted_rent:,.0f}</div>
            <div class="metric-sub">/ month</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">80% Confidence Range</div>
            <div class="metric-value" style="font-size:1.35rem; line-height:2.2rem;">₹{ci_lower:,.0f} - ₹{ci_upper:,.0f}</div>
            <div class="metric-sub">Random Forest Ensemble Spread</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Recommended Budget</div>
            <div class="metric-value" style="font-size:1.35rem; line-height:2.2rem;">₹{affordable_rent_30:,.0f} - ₹{affordable_rent_40:,.0f}</div>
            <div class="metric-sub">30% - 40% of Monthly Salary</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        if predicted_rent <= affordable_rent_30:
            badge_html = '<span class="badge-safe">✅ Healthy & Safe</span>'
            status_desc = "Well within 30% of income"
        elif predicted_rent <= affordable_rent_40:
            badge_html = '<span class="badge-warning">⚠️ Moderate Budget</span>'
            status_desc = "Between 30% and 40% of income"
        else:
            badge_html = '<span class="badge-danger">🚨 Budget Stretch</span>'
            status_desc = "Exceeds 40% of income"

        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Affordability Status</div>
            <div style="margin-top:8px;">{badge_html}</div>
            <div class="metric-sub" style="margin-top:10px;">{status_desc} ({rent_to_income_ratio:.1f}% of salary)</div>
        </div>
        """, unsafe_allow_html=True)

    # Budget Gauge Chart
    st.markdown("### 📊 Monthly Income Allocation")
    gauge_col, detail_col = st.columns([3, 2])

    with gauge_col:
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = rent_to_income_ratio,
            number = {'suffix': "%", 'valueformat': '.1f'},
            title = {'text': "Rent as % of Take-Home Salary"},
            gauge = {
                'axis': {'range': [0, 60], 'ticks': "outside"},
                'bar': {'color': "#21dc7d" if rent_to_income_ratio <= 35 else "#f59e0b" if rent_to_income_ratio <= 45 else "#ef4444"},
                'steps': [
                    {'range': [0, 30], 'color': "rgba(33, 220, 125, 0.2)"},
                    {'range': [30, 40], 'color': "rgba(245, 158, 11, 0.2)"},
                    {'range': [40, 60], 'color': "rgba(239, 68, 68, 0.2)"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 3},
                    'thickness': 0.75,
                    'value': 40
                }
            }
        ))
        fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with detail_col:
        st.markdown("#### 💼 Financial Commitment Breakdown")
        annual_rent = predicted_rent * 12
        security_deposit = predicted_rent * 2.5
        remaining_monthly = salary - predicted_rent

        st.write(f"• **Estimated Annual Rent**: `₹{annual_rent:,.0f}`")
        st.write(f"• **Estimated Security Deposit (2.5 mo)**: `₹{security_deposit:,.0f}`")
        st.write(f"• **Estimated Disposable Post-Rent Income**: `₹{remaining_monthly:,.0f} / mo`")
        st.info("💡 **Pro Tip**: In metro cities like Mumbai and Bangalore, landlords typically request 2–3 months security deposit for residential apartments.")

    st.markdown("---")

    # Localities in City matching Budget
    st.subheader(f"📍 Top Recommended Localities in {selected_city}")
    matching_df = df[
        (df['City'] == selected_city) &
        (df['BHK'] == selected_bhk) &
        (df['Rent'] >= affordable_rent_30 * 0.8) &
        (df['Rent'] <= affordable_rent_40 * 1.15)
    ].sort_values('Rent').drop_duplicates(subset=['Area Locality'])

    if not matching_df.empty:
        st.dataframe(
            matching_df[['Area Locality', 'Rent', 'Size', 'Price_per_SqFt', 'Furnishing Status', 'Tenant Preferred']].head(12),
            column_config={
                "Rent": st.column_config.NumberColumn("Rent (₹)", format="₹%d"),
                "Size": st.column_config.NumberColumn("Area (Sq. Ft.)", format="%d sq ft"),
                "Price_per_SqFt": st.column_config.NumberColumn("Price / Sq. Ft.", format="₹%.2f")
            },
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info(f"No exact listings matched the criteria in {selected_city} for a {selected_bhk} BHK in that range. Try exploring the Locality Explorer tab!")

# ==========================================
# TAB 2: MARKET ANALYTICS & TRENDS
# ==========================================
with tab2:
    st.subheader("📈 Real Estate Rental Market Insights")
    st.caption("Aggregated analytics based on ~4,700 real-world listings across 6 major Indian metropolitan cities.")

    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        # Average Rent by City
        city_avg = df.groupby('City')['Rent'].mean().reset_index().sort_values('Rent', ascending=False)
        fig_city = px.bar(
            city_avg,
            x='City',
            y='Rent',
            color='Rent',
            color_continuous_scale='Mint',
            title="Average Monthly Rent by City (in ₹)",
            text_auto='.0f'
        )
        fig_city.update_layout(xaxis_title="Metropolitan City", yaxis_title="Average Rent (₹)", coloraxis_showscale=False)
        st.plotly_chart(fig_city, use_container_width=True)

    with row1_col2:
        # Rent vs BHK Boxplot
        fig_bhk = px.box(
            df[df['BHK'] <= 4],
            x='BHK',
            y='Rent',
            color='BHK',
            title="Rent Distribution by BHK Configuration",
            points="outliers"
        )
        fig_bhk.update_layout(xaxis_title="Number of BHKs", yaxis_title="Monthly Rent (₹)", showlegend=False)
        st.plotly_chart(fig_bhk, use_container_width=True)

    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        # Furnishing Status Comparison
        furnish_avg = df.groupby(['City', 'Furnishing Status'])['Rent'].mean().reset_index()
        fig_furn = px.bar(
            furnish_avg,
            x='City',
            y='Rent',
            color='Furnishing Status',
            barmode='group',
            title="Rent Comparison by Furnishing Status Across Metros"
        )
        fig_furn.update_layout(yaxis_title="Average Rent (₹)")
        st.plotly_chart(fig_furn, use_container_width=True)

    with row2_col2:
        # Size vs Rent Scatter with Trendline
        fig_scatter = px.scatter(
            df.sample(min(800, len(df)), random_state=42),
            x='Size',
            y='Rent',
            color='City',
            hover_data=['BHK', 'Area Locality'],
            title="Property Size (Sq. Ft.) vs. Monthly Rent"
        )
        fig_scatter.update_layout(xaxis_title="Size (Sq. Ft.)", yaxis_title="Rent (₹)")
        st.plotly_chart(fig_scatter, use_container_width=True)

# ==========================================
# TAB 3: MODEL BENCHMARKS & EXPLAINABILITY
# ==========================================
with tab3:
    st.subheader("🤖 Machine Learning Model Benchmarking & Evaluation")
    st.caption("Comparison of candidate regression algorithms trained on 80% train and 20% held-out test splits.")

    # Convert model comparison to DataFrame
    comp_df = pd.DataFrame(MODEL_COMPARISON).T.reset_index().rename(columns={'index': 'Model'})
    comp_df = comp_df.sort_values('R2', ascending=False)

    col_t1, col_t2 = st.columns([3, 2])

    with col_t1:
        st.markdown("#### 🏆 Candidate Models Benchmark Table")
        st.dataframe(
            comp_df,
            column_config={
                "Model": st.column_config.TextColumn("Algorithm"),
                "MAE": st.column_config.NumberColumn("MAE (₹)", format="₹%.2f"),
                "RMSE": st.column_config.NumberColumn("RMSE (₹)", format="₹%.2f"),
                "R2": st.column_config.NumberColumn("R² Score", format="%.4f"),
                "TrainTime_s": st.column_config.NumberColumn("Training Time", format="%.3fs")
            },
            use_container_width=True,
            hide_index=True
        )

        fig_comp = px.bar(
            comp_df,
            x='Model',
            y='R2',
            color='Model',
            text='R2',
            title="Model Comparison: R² Score (Higher is Better)"
        )
        fig_comp.update_layout(yaxis_range=[0.5, 0.75], showlegend=False)
        st.plotly_chart(fig_comp, use_container_width=True)

    with col_t2:
        st.markdown("#### 🧠 Architectural Insights & Interview Q&A")
        st.markdown("""
        **Why Random Forest won:**
        * **Non-Linear Interactions:** Housing rent exhibits non-linear relationships (e.g., location premium multiplies with square footage). Linear models fail to capture this without manual interaction terms.
        * **Ensemble Variance Reduction:** Averaging 100 decorrelated decision trees reduces prediction variance significantly compared to a single decision tree.
        * **Outlier Robustness:** Tree splits are monotonic with respect to features, making them naturally resistant to extreme numerical outliers.
        """)

    st.markdown("---")

    # Feature Importance & Actual vs Predicted
    col_f1, col_f2 = st.columns(2)

    with col_f1:
        st.markdown("#### 🔍 Feature Importances (Random Forest Gini Impurity)")
        feat_df = pd.DataFrame(list(FEATURE_IMPORTANCE.items()), columns=['Feature', 'Importance'])
        feat_df = feat_df.sort_values('Importance', ascending=True).tail(8)

        fig_imp = px.bar(
            feat_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Top Features Influencing Rent Price",
            color='Importance',
            color_continuous_scale='Greens'
        )
        fig_imp.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_f2:
        st.markdown("#### 🎯 Actual vs. Predicted Rents (Test Set)")
        fig_parity = px.scatter(
            TEST_PREDS,
            x='Actual',
            y='Predicted',
            opacity=0.6,
            title="Actual vs Predicted Rent (₹) with Ideal Line",
            color_discrete_sequence=['#21dc7d']
        )
        max_val = max(TEST_PREDS['Actual'].max(), TEST_PREDS['Predicted'].max())
        fig_parity.add_trace(go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode='lines',
            name='Ideal (y = x)',
            line=dict(color='#ef4444', dash='dash')
        ))
        fig_parity.update_layout(xaxis_title="Actual Rent (₹)", yaxis_title="Predicted Rent (₹)")
        st.plotly_chart(fig_parity, use_container_width=True)

# ==========================================
# TAB 4: LOCALITY EXPLORER
# ==========================================
with tab4:
    st.subheader("🧭 Neighborhood & Locality Explorer")
    st.caption("Search and explore neighborhood pricing patterns across Indian cities.")

    filter_col1, filter_col2, filter_col3 = st.columns(3)
    with filter_col1:
        explorer_city = st.selectbox("Filter City", options=city_options, key="explorer_city")
    with filter_col2:
        max_budget = st.number_input("Max Budget (₹)", min_value=5000, max_value=100000, value=35000, step=2500)
    with filter_col3:
        search_locality = st.text_input("Search Locality Keyword (optional)", placeholder="e.g. Bandra, Whitefield, Salt Lake")

    city_data = df[(df['City'] == explorer_city) & (df['Rent'] <= max_budget)]
    if search_locality:
        city_data = city_data[city_data['Area Locality'].str.contains(search_locality, case=False, na=False)]

    # Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Available Listings", len(city_data))
    m2.metric("Average Rent in Selection", f"₹{city_data['Rent'].mean():,.0f}" if not city_data.empty else "N/A")
    m3.metric("Avg Price / Sq. Ft.", f"₹{city_data['Price_per_SqFt'].mean():,.2f}" if not city_data.empty else "N/A")

    st.markdown("### 📋 Matching Listings")
    st.dataframe(
        city_data[['Area Locality', 'BHK', 'Rent', 'Size', 'Price_per_SqFt', 'Furnishing Status', 'Tenant Preferred', 'Bathroom']].sort_values('Rent'),
        column_config={
            "Rent": st.column_config.NumberColumn("Rent (₹)", format="₹%d"),
            "Size": st.column_config.NumberColumn("Size (Sq. Ft.)", format="%d sq ft"),
            "Price_per_SqFt": st.column_config.NumberColumn("Price/SqFt", format="₹%.2f")
        },
        use_container_width=True,
        hide_index=True
    )

st.markdown("---")
st.markdown("<p style='text-align: center; color: #6b7280;'>Built with ❤️ by Aadarsh | Smart Rent Predictor • Machine Learning Real Estate Advisory</p>", unsafe_allow_html=True)