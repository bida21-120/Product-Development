import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import hashlib
import joblib
import xgboost as xgb
import numpy as np
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="AI Solution Analysis Dashboard")

# --- Define Sales Targets by Year ---
SALES_TARGETS = {
    2024: {
        "team": 60000000,
        "individual": 10000000
    },
    2025: {
        "team": 10000000,
        "individual": 2000000
    }
}

def evaluate_performance(value, year, level="team"):
    target = SALES_TARGETS.get(year, {}).get(level)
    if target is None:
        return "⚪ No Target"
    if value >= target:
        return "🟢 Excellent"
    elif value >= target * 0.7:
        return "🟡 Good"
    else:
        return "🔴 Needs Improvement"

# --- User Authentication Setup ---
users = {
    "Boemo Marumo": {
        "password": hashlib.sha256("admin123".encode()).hexdigest(),
        "role": "Sales Manager"
    },
    "Alice Josephs": {
        "password": hashlib.sha256("sales2025".encode()).hexdigest(),
        "sales_member": "Alice Josephs"
    },
    "Bob Havertz": {
        "password": hashlib.sha256("sales2024".encode()).hexdigest(),
        "sales_member": "Bob Havertz"
    },
    "Carlos Mainoo": {
        "password": hashlib.sha256("sales2023".encode()).hexdigest(),
        "sales_member": "Carlos Mainoo"
    },
    "Darshen Henry": {
        "password": hashlib.sha256("sales2022".encode()).hexdigest(),
        "sales_member": "Darshen Henry"
    },
    "Ethan Knowles": {
        "password": hashlib.sha256("sales2021".encode()).hexdigest(),
        "sales_member": "Ethan Knowles"
    }
}

def login(username, password):
    user = users.get(username)
    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    if user and (user["password"] == hashed_pw or 
                st.session_state.password_updates.get(username) == hashed_pw):
        return user
    return None

# --- Session State Handling ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.SALES_TARGETS = SALES_TARGETS.copy()
    st.session_state.original_SALES_TARGETS = SALES_TARGETS
    st.session_state.password_updates = {}
if "show_reset" not in st.session_state:
    st.session_state.show_reset = False

if not st.session_state.logged_in:
    st.title("AI-Solution Analysis Dashboard")
    
    if st.session_state.show_reset:
        st.subheader("🔐 Forgot Password")
        email = st.text_input("Enter your registered email")
        if st.button("Send Reset Link"):
            if email:
                st.success(f"Password reset instructions have been sent to **{email}**.")
                st.session_state.show_reset = False
            else:
                st.error("Please enter a valid email.")
        if st.button("Back to Login"):
            st.session_state.show_reset = False
        st.stop()
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("Login"):
            user = login(username, password)
            if user:
                st.success("Login successful!")
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.user_info = user
                st.rerun()
            else:
                st.error("Invalid username or password.")
    with col2:
        if st.button("Forgot Password?"):
            st.session_state.show_reset = True
            st.rerun()

    st.stop()

# --- Authenticated Content Below ---
st.subheader("📊 AI-Solution Analysis Dashboard")
st.markdown('<style>div.block-container{padding-top:0rem !important;padding-bottom: 1rem;} header, footer {visibility: hidden;} </style>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load(r"c:\Users\bida21-120\Downloads\xgboost.joblib")

model_bundle = load_model()
xgb_model = model_bundle['model']
preprocessor = model_bundle['preprocessor']

@st.cache_data
def load_data():
    df = pd.read_csv("AISolution_web_server_logs_final.csv", parse_dates=["timestamp"], dayfirst=True)
    df['date'] = df['timestamp'].dt.date
    df['year'] = df['timestamp'].dt.year
    df['hour'] = df['timestamp'].dt.hour
    return df

@st.cache_data
def preprocess_sessions(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by=['ip_address', 'timestamp'])
    
    df['time_diff'] = df.groupby('ip_address')['timestamp'].diff().dt.total_seconds().div(60)
    df['new_session'] = (df['time_diff'] > 30) | (df['time_diff'].isna())
    
    df['session_id'] = df['new_session'].cumsum().astype(int)
    
    return df

df = preprocess_sessions(load_data())

# --- Sidebar Filters ---
with st.sidebar:
    st.image(r"c:\Users\bida21-120\Downloads\ai-solution-high-resolution-logo.png", use_container_width=True)
    st.markdown("### 👤 User Info")
    st.write(f"**Username:** {st.session_state.username}")
    st.write(f"**Role:** {st.session_state.user_info.get('role', 'N/A')}")

    st.markdown("### 📂 Navigation")
    st.session_state.current_page = st.selectbox("Select Page", ["Dashboard", "Sales Forecast"], label_visibility= "collapsed")

st.sidebar.header("📅 Filter Data")
sales_members = df['sales_member'].unique().tolist()

# Only show all members if admin, otherwise default to the logged-in user
user_info = st.session_state.user_info
if user_info.get("role") == "Sales Manager":
    selected_member = st.sidebar.selectbox("Select Sales Member", sales_members)
    can_select_member = user_info.get("role") == "Sales Manager"
else:
    selected_member = user_info.get("sales_member")
    can_select_member = False

min_date, max_date = df['date'].min(), df['date'].max()
date_range = st.sidebar.date_input("Date Range", [min_date, max_date])

# Updated country filter with "All" option
country_options = ['All'] + sorted(df['country'].unique().tolist())
selected_countries = st.sidebar.multiselect(
    "Country", 
    options=country_options,
    default=['All'],
    help="Select countries to filter by"
)
# Handle "All" selection
if 'All' in selected_countries:
    country_filter = df['country'].unique().tolist()
else:
    country_filter = selected_countries

# Updated product filter with "All" option
product_options = ['All'] + sorted(df['product_type'].unique().tolist())
selected_products = st.sidebar.multiselect(
    "Product Type", 
    options=product_options,
    default=['All'],
    help="Select product types to filter by"
)

# Handle "All" selection
if 'All' in selected_products:
    product_filter = df['product_type'].unique().tolist()
else:
    product_filter = selected_products

# --- Logout Button ---
with st.sidebar:
    if st.session_state.get("logged_in", False):
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.rerun()

# --- Apply Filters ---
df_filtered = df[
    (df['date'] >= date_range[0]) &
    (df['date'] <= date_range[1]) &
    (df['country'].isin(country_filter)) &
    (df['product_type'].isin(product_filter))
]

df_sales = df_filtered[df_filtered['status_code'] == 200]

total_sessions = df_filtered['session_id'].nunique()
demo_sessions = df_filtered[df_filtered['interaction_type'] == 'demo']['session_id'].nunique()
purchase_sessions = df_filtered[(df_filtered['status_code'] == 200) & (df_filtered['method'] == 'POST')]['session_id'].nunique()
demo_rate = (demo_sessions / total_sessions) * 100 if total_sessions else 0
purchase_rate = (purchase_sessions / total_sessions) * 100 if total_sessions else 0

# Add this after loading and preprocessing
df_filtered['year'] = pd.DatetimeIndex(df_filtered['timestamp']).year

# Determine which years are in the filtered data
active_years = sorted(df_filtered['year'].unique())

cols = st.columns(len(active_years))  # Create a column for each year

total_revenue = df_sales['price'].sum()

# Define Targets
daily_target = 100000
product_target = 5000
country_target = 10000
conversion_target = 0.20

if st.session_state.current_page == "Dashboard":
    # --- Tabs ---
    tab1, tab2 = st.tabs(["Team Sales Overview", "Individual Performance"])

    # ========== TAB 1: TEAM OVERVIEW ==========
    with tab1:
        col1, col2, col3 = st.columns([1.5, 1.5, 0.8])

        with col1:
            st.markdown("**📈 Sales Over Time**")
            rev_over_time = df_filtered[df_filtered['is_sale'] == 1].groupby("date")["price"].sum().reset_index()
            fig_rev = go.Figure()
            fig_rev.add_trace(go.Scatter(x=rev_over_time["date"], y=rev_over_time["price"], mode="lines+markers", name="Actual Revenue", line=dict(color="blue", width=3)))
            fig_rev.add_trace(go.Scatter(x=rev_over_time["date"], y=[daily_target]*len(rev_over_time), mode="lines", name="Target Revenue", line=dict(color="green", dash="dash")))
            fig_rev.update_layout(height=210, yaxis_title="Revenue (£)", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_rev, use_container_width=True)
            st.markdown(f"✅ **{(rev_over_time['price'] > daily_target).sum()} days** met or exceeded the revenue target of £{daily_target:,}.")

            st.markdown("**🔄 User Journey Funnel: Views → Demo → Sales**")
            views = df_filtered[
            (df_filtered['method'].isin(['GET', 'POST'])) & 
            (df_filtered['status_code'] == 200) & 
            (df_filtered['interaction_type'] == 'product')
            ].shape[0]

            demos = df_filtered[df_filtered['interaction_type'] == 'demo'].shape[0]
            sales = df_filtered['is_sale'].sum()

            # --- Define performance targets ---
            targets = {
                'Total Views': 3000,
                'Demos': 1000,
                'Purchases': 500
            }

            # --- Funnel data dictionary ---
            funnel_stages = {
                'Demos': demos,
                'Purchases': sales,
                'Total Views': views   
            }

            # --- Plotly funnel chart ---
            funnel_df = pd.DataFrame({
                'Stage': list(funnel_stages.keys()),
                'Count': list(funnel_stages.values())
            })

            fig_funnel = px.funnel(
                funnel_df,
                x='Count',
                y='Stage',
                height=210,
                color='Stage',
                color_discrete_sequence=px.colors.qualitative.Pastel
            )

            fig_funnel.update_layout(
                margin=dict(l=0, r=0, t=30, b=0),
                showlegend=False
            )

            st.plotly_chart(fig_funnel, use_container_width=True)

        with col2:
            df_member = df_sales[df_sales['sales_member'] == selected_member]
            product_targets = {
                'AI-powered virtual assistant': 5000,
                'Intelligent CRM': 4000,
                'Predictive Analytics Tool': 4500,
                'Chatbot Solution': 3000,
                'Smart Recommender System': 3500
            }
            st.markdown("**🛍️ Sales by Product Type**")

            # Calculate product sales and performance
            product_sales = df_filtered[df_filtered['is_sale'] == 1].groupby("product_type")["price"].sum().reset_index()
            product_sales["Performance"] = product_sales["price"].apply(lambda x: "✅ Hit" if x >= product_target else "❌ Miss")

            # Create line chart with color-coded markers
            fig_prod = px.line(
                product_sales, 
                x="product_type", 
                y="price",
                markers=True,  # Show markers on lines
                color="Performance",
                color_discrete_map={"✅ Hit": "green", "❌ Miss": "red"},
                text="price"  # Show values on markers
            )

            # Add target reference line
            fig_prod.add_hline(
                y=product_target,
                line_dash="dot",
                line_color="orange",
                annotation_text=f"Target:",
                annotation_position="top right"
            )

            # Customize layout
            fig_prod.update_layout(
                height=210,
                yaxis_title="Sales (£)",
                xaxis_title="Product Type",
                margin=dict(l=0, r=0, t=30, b=0),
                hovermode="x unified"
            )

            # Add explanatory note in Streamlit
            st.markdown("""
            <div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
                <span style="color: green;">✅ Green</span> = Exceeded sales target<br>
                <span style="color: red;">❌ Red</span> = Below target
            </div>
            """, unsafe_allow_html=True)

            st.plotly_chart(fig_prod, use_container_width=True)

            st.markdown("**🌍 Sales Distribution by Country**")
            # Filter for only sales
            sales_df = df_filtered[df_filtered['is_sale'] == 1]
            # Aggregate sales by country
            country_sales = sales_df.groupby('country')['price'].sum().reset_index().sort_values(by='price', ascending=True)
            # Plot
            fig_map = px.choropleth(
                country_sales,
                locations='country',
                locationmode='country names',
                color='price',
                color_continuous_scale='Blues',
                labels={'price': 'Target (£)'},
                title="Sales by Country"
            )
            fig_map.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=210)
            st.plotly_chart(fig_map, use_container_width=True)

        with col3:
            st.markdown("**📈 Conversion Metrics**")
            st.metric("💵 Total Revenue", f"£{total_revenue:,.2f}")
            active_years = sorted(df_filtered['year'].unique())
            for i, year in enumerate(active_years):
                st.markdown(f"**📅 {year}**")
                df_year = df_filtered[df_filtered['year'] == year]
                df_sales_year = df_year[df_year['status_code'] == 200]
                year_revenue = df_sales_year['price'].sum()
                team_perf = evaluate_performance(year_revenue, year, level="team")
                st.metric(label=f"Team Revenue", value=f"£{year_revenue:,.2f}", delta=team_perf)

    # ========== TAB 2: INDIVIDUAL PERFORMANCE ==========
    with tab2:

        view_option = st.selectbox(
            "Select View",
            ["Performance Dashboard", "Team Comparison"],
            key="individual_view"
        )
        # Only show header if NOT in Team Comparison view
        if view_option != "Team Comparison":
            if can_select_member:
                st.markdown(f"### 👤 Individual Sales Performance for {selected_member}")
            else:
                st.markdown("### 👤 Your Sales Performance")

        if view_option == "Performance Dashboard":
            df_member = df_sales[df_sales['sales_member'] == selected_member]
            df_member_interactions = df_filtered[df_filtered['sales_member'] == selected_member]

        
            individual_targets = {
                "Darshen Henry": {2024: 90000000, 2025: 10000000},
                "Alice Josephs": {2024: 93000000, 2025: 20000000},
                "Carlos Mainoo": {2024: 1000000, 2025: 12500000},
                "Ethan Knowles": {2024: 950000, 2025: 10500000},
                "Bob Havertz": {2024: 1050000, 2025: 13000000},
            }

            # Determine target based on selected member and year
            current_year = pd.to_datetime(df_member["date"].max()).year if not df_member.empty else datetime.now().year
            annual_target = individual_targets.get(selected_member, {}).get(current_year, 0)
            daily_target = round(annual_target / 365, 2)
            days_in_period = (date_range[1] - date_range[0]).days + 1
            period_target = (annual_target / 365) * days_in_period

            col1, col2, col3 = st.columns([1.5, 1.5, 0.8])

            with col1:
                st.markdown("**Sales Over Time**")
                member_rev = df_member.groupby('date')['price'].sum().reset_index()
        
                fig_member_rev = go.Figure()
                fig_member_rev.add_trace(go.Scatter(
                    x=member_rev["date"],
                    y=member_rev["price"],
                    mode="lines+markers",
                    name="Actual Revenue",
                    line=dict(color="orange", width=3)
                ))
                fig_member_rev.add_trace(go.Scatter(
                    x=member_rev["date"],
                    y=[daily_target]*len(member_rev),
                    mode="lines",
                    name="Target Revenue",
                    line=dict(color="green", dash="dash")
                ))
                fig_member_rev.update_layout(
                    height=210,
                    yaxis_title="Sales (£)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig_member_rev, use_container_width=True)

                st.markdown(f"✅ **{(member_rev['price'] > daily_target).sum()} days** met or exceeded the revenue target of £{daily_target:,}.")

                st.markdown("**🏆 Top Products Sold**")

                # Aggregate and sort top 5 products
                top_products = (
                    df_member[df_member['is_sale'] == 1]  # ensure only real sales
                    .groupby("product_type")["price"]
                    .sum()
                    .reset_index()
                    .sort_values(by="price", ascending=True)  # for horizontal bar ordering
                    .tail(5)  # top 5
                )

                # Plot with continuous color scale
                fig_top_products = px.bar(
                    top_products,
                    x="price",
                    y="product_type",
                    orientation="h",
                    labels={"price": "Target (£)", "product_type": "Product"},
                    color="price",
                    color_continuous_scale="Blues"
                )

                fig_top_products.update_layout(
                    height=210,
                    margin=dict(l=0, r=0, t=30, b=0),
                    coloraxis_showscale=True,  # hide color legend for clean look
                )

                st.plotly_chart(fig_top_products, use_container_width=True)

            with col2:
                st.markdown("**Sales by Country**")
                country_breakdown = df_member.groupby('country')['price'].sum().reset_index()
            # Define example targets per country
                country_targets = {
                    'United Kingdom': 300000,
                    'Ireland': 250000,
                    'France': 200000,
                    'Belgium': 180000,
                    'Netherlands': 150000,
                    'Germany': 220000,
                    'Norway': 200000
                }

            # Merge targets into DataFrame
                country_breakdown['target'] = country_breakdown['country'].map(country_targets)
                country_breakdown['met_target'] = country_breakdown['price'] >= country_breakdown['target']

            # Plot with color-coded bars
                fig_country = px.bar(
                    country_breakdown,
                    x='country',
                    y='price',
                    color='met_target',
                    color_discrete_map={True: 'green', False: 'orange'},
                    height=210
                )
                fig_country.update_layout(
                    yaxis_title="Sales (£)",
                    legend_title_text="Target Met",
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                st.plotly_chart(fig_country, use_container_width=True, key="bar_country")

                st.markdown("**🎯 Conversion Rate**")

            # Calculate conversion rate
                total_interactions = len(df_member)
                total_sales = df_member["is_sale"].sum()
                conversion_rate = (total_sales / total_interactions) * 100 if total_interactions > 0 else 0

            # Gauge chart
                fig_conv = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=conversion_rate,
                    domain={"x": [0, 1], "y": [0, 0.95]},
                    title={"text": "Conversion Rate", "font": {"size": 12}},
                    number={"suffix": "%", "font": {"size": 24}, "valueformat": ".2f"},
                    delta={"reference": 75, "increasing": {"color": "green"}, "decreasing": {"color": "red"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "darkgray"},
                        "bar": {"color": "mediumseagreen"},
                        "steps": [
                            {"range": [0, 25], "color": "#ffcccc"},
                            {"range": [25, 50], "color": "#ffe680"},
                            {"range": [50, 75], "color": "#d4f7dc"},
                            {"range": [75, 100], "color": "#b3ffb3"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 75
                        }
                    }
                ))

                fig_conv.update_layout(
                    height=210,
                    margin=dict(l=0, r=0, t=20, b=10),
                )

                st.plotly_chart(fig_conv, use_container_width=True)

            with col3:
                st.markdown("**💼 Performance Metrics**")
                avg_sale = df_member['price'].mean()
                st.metric("💰 Average Sale", f"£{avg_sale:,.2f}" if not pd.isna(avg_sale) else "N/A")

                df_member['year'] = pd.DatetimeIndex(df_member['timestamp']).year
                member_years = sorted(df_member['year'].unique())

                for year in member_years:
                    df_member_year = df_member[df_member['year'] == year]
                    member_rev = df_member_year['price'].sum()
                    ind_perf = evaluate_performance(member_rev, year, level="individual")
                    st.metric(f"💰 Revenue ({year})", f"£{member_rev:,.2f}", delta=ind_perf)

                if not df_member.empty:
                    top_product = df_member.groupby('product_type')['price'].sum().idxmax()
                    st.metric("🏆 Top Product", top_product)
                else:
                    st.metric("🏆 Top Product", "N/A")
        
        elif view_option == "Team Comparison":
            st.markdown("**🏆 Team Performance Comparison**")
        
        # Calculate metrics for all team members
           # Calculate metrics
            comparison_df = pd.DataFrame({
                'Member': sales_members,
                'Total Sales': [df_sales[df_sales['sales_member'] == m]['price'].sum() for m in sales_members],
                'Avg Sale': [df_sales[df_sales['sales_member'] == m]['price'].mean() for m in sales_members],
                'Deals Closed': [df_sales[df_sales['sales_member'] == m].shape[0] for m in sales_members],
                'Conversion Rate': [
                    (df_sales[df_sales['sales_member'] == m].shape[0] /
                    df_filtered[df_filtered['sales_member'] == m].shape[0]) * 100
                    if df_filtered[df_filtered['sales_member'] == m].shape[0] > 0 else 0
                    for m in sales_members
                ]
            })

            # Highlight selected member
            def highlight_row(row):
                return ['background-color: #e6f3ff'] * len(row) if row['Member'] == selected_member else [''] * len(row)

            # Apply styling
            styled_df = comparison_df.style\
                .apply(highlight_row, axis=1)\
                .format({
                    'Total Sales': '£{:,.2f}',
                    'Avg Sale': '£{:,.2f}',
                    'Conversion Rate': '{:.1f}%'
                })\
                .background_gradient(cmap='Blues', subset=['Total Sales', 'Avg Sale'])

            # Display with scrollable wrapper
            st.markdown("""
                <style>
                .scrollable-table {
                    height: 210px;
                    overflow-y: auto;
                    border: 1px solid #ddd;
                    border-radius: 6px;
                }
                </style>
            """, unsafe_allow_html=True)

            st.markdown(f"<div class='scrollable-table'>{styled_df.to_html(escape=False)}</div>", unsafe_allow_html=True)
                      
        # Add some comparative charts
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Total Sales Comparison**")
                fig = px.bar(
                    comparison_df.sort_values('Total Sales', ascending=True),
                    x='Total Sales',
                    y='Member',
                    orientation='h',
                    color='Total Sales',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(height=210, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig, use_container_width=True)
        
            with col2:
                st.markdown("**Conversion Rate Comparison**")
                fig = px.bar(
                    comparison_df.sort_values('Conversion Rate', ascending=True),
                    x='Conversion Rate',
                    y='Member',
                    orientation='h',
                    color='Conversion Rate',
                    color_continuous_scale='Tealrose'
                )
                fig.update_layout(height=210, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig, use_container_width=True)

elif st.session_state.current_page == "Sales Forecast":
    st.markdown("**Accuracy:0.82**")
    st.markdown("**Predict Conversion Likelihood**")

    with st.form("conversion_form"):
        method = st.selectbox("HTTP Method", ["GET", "POST"])
        country = st.selectbox("Country", sorted(df["country"].unique()))
        interaction_type = st.selectbox("Interaction Type", sorted(df["interaction_type"].unique()))
        product_type = st.selectbox("Product Type", sorted(df["product_type"].unique()))
        price = st.number_input("Price (£)", min_value=0.0, value=50.0)
        status_code = st.number_input("Status Code", min_value=100, max_value=599, value=200)
        response_size = st.number_input("Response Size (bytes)", min_value=0, value=1000)

        submitted = st.form_submit_button("Predict")

    if submitted:
        # Create a single-row input DataFrame
        input_data = pd.DataFrame([{
            'method': method,
            'country': country,
            'interaction_type': interaction_type,
            'product_type': product_type,
            'price': price,
            'status_code': status_code,
            'response_size': response_size
        }])
        try:

            # Preprocess the input using saved preprocessor
            input_encoded = preprocessor.transform(input_data)
            if callable(input_encoded): raise ValueError("Preprocessor returned a function instead of data")
            dmatrix_input = xgb.DMatrix(input_encoded)

            # Predict using XGBoost model
            prediction = xgb_model.predict(dmatrix_input)[0]
            is_conversion = bool(prediction > 0.5)

            st.subheader("Prediction Result")
            st.write(f"**Conversion Probability:** {prediction:.2%}")
            
            if is_conversion:
                st.success("Likely to Convert") 
            else:
                st.warning("Unlikely to Convert")

            # Optional: Show the processed input data for debugging
            with st.expander("Show processed input"):
                st.dataframe(input_encoded)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please check your input data and try again.")