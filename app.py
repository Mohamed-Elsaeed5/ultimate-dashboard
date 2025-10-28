import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Sleep Health Analysis Dashboard",
    page_icon="😴",
    layout="wide",
    initial_sidebar_state="expanded"
)
# place this near the top of your app (after st.set_page_config)

with st.sidebar:
    st.image("insomnia.png", caption=None, use_container_width=True)
    st.markdown("---")           # optional line separator
    st.subheader("Navigation & Filters")

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    h1 {
        color: #1f77b4;
        text-align: center;
    }
    h2 {
        color: #2c3e50;
    }
    </style>
""", unsafe_allow_html=True)

# Load Data Function
@st.cache_data
def load_data():
    """Load and preprocess the sleep health data"""
    df = pd.read_csv('sleep.csv')
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Handle missing values
    df['Heart Rate'] = df['Heart Rate'].fillna(df['Heart Rate'].median())
    df['Blood Pressure'] = df['Blood Pressure'].fillna(df['Blood Pressure'].mode()[0])
    df['Occupation'] = df['Occupation'].fillna('Unknown')
    df['Sleep Duration'] = df['Sleep Duration'].fillna(df['Sleep Duration'].median())
    df['Sleep Disorder'] = df['Sleep Disorder'].fillna('Normal')
    # df['BMI Category']=df['BMI Category'].replace('Normal Weight','Normal')
    # Split Blood Pressure into Systolic and Diastolic
    df[['Systolic', 'Diastolic']] = df['Blood Pressure'].str.split('/', expand=True)
    df['Systolic'] = pd.to_numeric(df['Systolic'], errors='coerce')
    df['Diastolic'] = pd.to_numeric(df['Diastolic'], errors='coerce')
    
    return df

# Load the data
df = load_data()

# Sidebar - Navigation and Filters
st.sidebar.title("🎯 Navigation & Filters")
page = st.sidebar.radio(
    "Select Analysis Page:",
    ["🏠 Overview", "📊 Exploratory Analysis", "🔍 Deep Dive Analysis", 
     "📈 Statistical Analysis", "🎯 Predictive Insights", "📋 Data Explorer"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Filters")

# Filters
age_range = st.sidebar.slider(
    "Age Range:",
    int(df['Age'].min()),
    int(df['Age'].max()),
    (int(df['Age'].min()), int(df['Age'].max()))
)

gender_filter = st.sidebar.multiselect(
    "Gender:",
    options=df['Gender'].unique(),
    default=df['Gender'].unique()
)

occupation_filter = st.sidebar.multiselect(
    "Occupation:",
    options=sorted(df['Occupation'].unique()),
    default=sorted(df['Occupation'].unique())
)

disorder_filter = st.sidebar.multiselect(
    "Sleep Disorder:",
    options=df['Sleep Disorder'].unique(),
    default=df['Sleep Disorder'].unique()
)

# Apply filters
filtered_df = df[
    (df['Age'] >= age_range[0]) & 
    (df['Age'] <= age_range[1]) &
    (df['Gender'].isin(gender_filter)) &
    (df['Occupation'].isin(occupation_filter)) &
    (df['Sleep Disorder'].isin(disorder_filter))
]

st.sidebar.markdown("---")
st.sidebar.info(f"**Filtered Records:** {len(filtered_df)} / {len(df)}")

# Download filtered data
@st.cache_data
def convert_df_to_csv(dataframe):
    return dataframe.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(filtered_df)
st.sidebar.download_button(
    label="📥 Download Filtered Data",
    data=csv,
    file_name='filtered_sleep_data.csv',
    mime='text/csv',
)

# Main Content Area
if page == "🏠 Overview":
    st.title("😴 Sleep Health & Lifestyle Analysis Dashboard")
    st.markdown("### Comprehensive Analysis of Sleep Patterns and Health Metrics")
   #for card color 
    st.markdown("""
<style>
/* KPI card */
[data-testid="stMetric"] {
  background: rgba(255,255,255,0.06);   /* optional: subtle card bg on dark theme */
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 12px;
  padding: 14px 16px;
}

/* label (small text above) */
[data-testid="stMetricLabel"] > div {
  color: #ffffff !important;
  opacity: 0.85;                         /* keep a bit dimmer than value */
}

/* main value */
[data-testid="stMetricValue"] {
  color: #ffffff !important;
}

/* delta text + badge */
[data-testid="stMetricDelta"] {
  color: #ffffff !important;             /* make delta text white */
}
[data-testid="stMetricDelta"] svg {      /* make the arrow white too */
  filter: brightness(0) invert(1);
}

/* optional: make all plotly charts stretch instead of using use_container_width */
.user-select-none svg { max-width: 100%; }
</style>
""", unsafe_allow_html=True)

    # Key Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="👥 Total Participants",
            value=f"{len(filtered_df):,}",
            delta=f"{len(filtered_df) - len(df)}" if len(filtered_df) != len(df) else None
        )
    
    with col2:
        avg_sleep = filtered_df['Sleep Duration'].mean()
        st.metric(
            label="😴 Avg Sleep Duration",
            value=f"{avg_sleep:.1f} hrs",
            delta=f"{avg_sleep - 7:.1f} hrs" if avg_sleep != 7 else "Optimal"
        )
    
    with col3:
        avg_quality = filtered_df['Quality of Sleep'].mean()
        st.metric(
            label="⭐ Avg Sleep Quality",
            value=f"{avg_quality:.1f}/10",
            delta=f"{avg_quality - 7:.1f}" if avg_quality != 7 else "Good"
        )
    
    with col4:
        disorder_pct = (filtered_df['Sleep Disorder'] != 'None').sum() / len(filtered_df) * 100
        st.metric(
            label="⚠️ With Sleep Disorders",
            value=f"{disorder_pct:.1f}%",
            delta=f"{disorder_pct:.1f}%" if disorder_pct > 0 else "None"
        )
    
    with col5:
        avg_steps = filtered_df['Daily Steps'].mean()
        st.metric(
            label="👣 Avg Daily Steps",
            value=f"{avg_steps:,.0f}",
            delta="Healthy" if avg_steps >= 7000 else "Low"
        )
    
    st.markdown("---")
    
    # Overview Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sleep Disorder Distribution")
        disorder_counts = filtered_df['Sleep Disorder'].value_counts()
        fig = px.pie(
            values=disorder_counts.values,
            names=disorder_counts.index,
            title="Sleep Disorder Breakdown",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Age Distribution")
        fig = px.histogram(
            filtered_df,
            x='Age',
            nbins=5,
            title="Participant Age Distribution",
            color_discrete_sequence=["#651fb4"]
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Gender Distribution by Occupation")
        gender_occ = filtered_df.groupby(['Occupation', 'Gender']).size().reset_index(name='Count')
        fig = px.bar(
            gender_occ,
            x='Occupation',
            y='Count',
            color='Gender',
            title="Occupation by Gender",
            barmode='group',
            color_discrete_map={'Male': '#3498db', 'Female': '#e74c3c'}
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("BMI Category Distribution")
        bmi_counts = filtered_df['BMI Category'].value_counts()
        fig = px.bar(
            x=bmi_counts.index,
            y=bmi_counts.values,
            title="BMI Category Distribution",
            color=bmi_counts.values,
            color_continuous_scale='Blues'
        )
        fig.update_layout(showlegend=False, xaxis_title="BMI Category", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

elif page == "📊 Exploratory Analysis":
    st.title("📊 Exploratory Data Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Sleep Patterns", "Health Metrics", "Lifestyle Factors", "Correlations"])
    
    with tab1:
        st.subheader("Sleep Pattern Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sleep Duration by Occupation
            sleep_by_occ = filtered_df.groupby('Occupation')['Sleep Duration'].mean().sort_values(ascending=False)
            fig = px.bar(
                x=sleep_by_occ.values,
                y=sleep_by_occ.index,
                orientation='h',
                title="Average Sleep Duration by Occupation",
                labels={'x': 'Hours', 'y': 'Occupation'},
                color=sleep_by_occ.values,
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Sleep Quality by Age Group
            filtered_df['Age_Group'] = pd.cut(filtered_df['Age'], bins=[0, 30, 40, 50, 100], labels=['<30', '30-40', '40-50', '50+'])
            quality_by_age = filtered_df.groupby('Age_Group')['Quality of Sleep'].mean()
            fig = px.line(
                x=quality_by_age.index.astype(str),
                y=quality_by_age.values,
                title="Sleep Quality by Age Group",
                markers=True,
                labels={'x': 'Age Group', 'y': 'Quality Score'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Sleep Duration vs Quality Scatter
        fig = px.scatter(
            filtered_df,
            x='Sleep Duration',
            y='Stress Level',
            color='Sleep Disorder',
            size='Stress Level',
            hover_data=['Occupation', 'Age', 'Gender'],
            title="Sleep Duration vs Quality (size = Stress Level)",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Health Metrics Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Blood Pressure Distribution
            fig = go.Figure()
            fig.add_trace(go.Box(y=filtered_df['Systolic'], name='Systolic', marker_color='#e74c3c'))
            fig.add_trace(go.Box(y=filtered_df['Diastolic'], name='Diastolic', marker_color='#3498db'))
            fig.update_layout(title="Blood Pressure Distribution", yaxis_title="mmHg")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Heart Rate by Sleep Disorder
            fig = px.box(
                filtered_df,
                x='Sleep Disorder',
                y='Heart Rate',
                color='Sleep Disorder',
                title="Heart Rate by Sleep Disorder",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # BMI vs Sleep Quality
        col1, col2 = st.columns(2)
        
        with col1:
            bmi_sleep = filtered_df.groupby('BMI Category')['Sleep Duration'].mean().sort_values()
            fig = px.bar(
                x=bmi_sleep.index,
                y=bmi_sleep.values,
                title="Average Sleep Duration by BMI Category",
                color=bmi_sleep.values,
                color_continuous_scale='RdYlGn_r'
            )
            fig.update_layout(xaxis_title="BMI Category", yaxis_title="Hours")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.violin(
                filtered_df,
                x='BMI Category',
                y='Heart Rate',
                color='BMI Category',
                title="Heart Rate Distribution by BMI Category",
                box=True
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Lifestyle Factors Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Physical Activity vs Sleep Quality (removed trendline)
            fig = px.scatter(
                filtered_df,
                x='Physical Activity Level',
                y='Quality of Sleep',
                color='Sleep Disorder',
                title="Physical Activity vs Sleep Quality",
                labels={'Physical Activity Level': 'Activity Level (min/day)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Stress Level Distribution
            stress_counts = filtered_df['Stress Level'].value_counts().sort_index()
            fig = px.bar(
                x=stress_counts.index,
                y=stress_counts.values,
                title="Stress Level Distribution",
                labels={'x': 'Stress Level (1-10)', 'y': 'Count'},
                color=stress_counts.values,
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Daily Steps Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            steps_sleep = filtered_df.groupby(pd.cut(filtered_df['Daily Steps'], bins=5))['Sleep Duration'].mean()
            fig = px.line(
                x=[str(i) for i in steps_sleep.index],
                y=steps_sleep.values,
                title="Sleep Duration by Daily Steps Range",
                markers=True
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(
                filtered_df,
                x='Daily Steps',
                y='Physical Activity Level',
                color='Sleep Disorder',
                size='Quality of Sleep',
                title="Daily Steps vs Physical Activity",
                hover_data=['Occupation']
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Correlation Analysis")
        
        # Correlation Heatmap
        numeric_cols = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 
                       'Stress Level', 'Heart Rate', 'Daily Steps', 'Systolic', 'Diastolic']
        corr_matrix = filtered_df[numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect='auto',
            color_continuous_scale='RdBu_r',
            title="Correlation Matrix of Health Metrics"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Pairplot alternative - scatter matrix
        st.subheader("Scatter Plot Matrix")
        selected_vars = st.multiselect(
            "Select variables for scatter matrix:",
            numeric_cols,
            default=['Sleep Duration', 'Quality of Sleep', 'Stress Level', 'Physical Activity Level']
        )
        
        if len(selected_vars) >= 2:
            fig = px.scatter_matrix(
                filtered_df,
                dimensions=selected_vars,
                color='Sleep Disorder',
                title="Scatter Plot Matrix"
            )
            fig.update_traces(diagonal_visible=False)
            st.plotly_chart(fig, use_container_width=True)

elif page == "🔍 Deep Dive Analysis":
    st.title("🔍 Deep Dive Analysis")
    
    analysis_type = st.selectbox(
        "Select Analysis Type:",
        ["Occupation Analysis", "Gender Analysis", "Age Group Analysis", "Sleep Disorder Analysis"]
    )
    
    if analysis_type == "Occupation Analysis":
        st.subheader("Detailed Occupation Analysis")
        
        # Occupation metrics table
        occ_metrics = filtered_df.groupby('Occupation').agg({
            'Person ID': 'count',
            'Sleep Duration': 'mean',
            'Quality of Sleep': 'mean',
            'Stress Level': 'mean',
            'Physical Activity Level': 'mean',
            'Daily Steps': 'mean',
            'Heart Rate': 'mean'
        }).round(2)
        occ_metrics.columns = ['Count', 'Avg Sleep (hrs)', 'Avg Quality', 'Avg Stress', 
                               'Avg Activity', 'Avg Steps', 'Avg Heart Rate']
        
        st.dataframe(occ_metrics.style.background_gradient(cmap='YlOrRd'), use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sleep disorder prevalence by occupation
            disorder_occ = pd.crosstab(filtered_df['Occupation'], filtered_df['Sleep Disorder'], normalize='index') * 100
            fig = px.bar(
                disorder_occ,
                title="Sleep Disorder Prevalence by Occupation (%)",
                barmode='stack',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # BMI distribution by occupation
            fig = px.box(
                filtered_df,
                x='Occupation',
                y='Sleep Duration',
                color='BMI Category',
                title="Sleep Duration Distribution by Occupation & BMI"
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Gender Analysis":
        st.subheader("Detailed Gender Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gender comparison metrics
            gender_metrics = filtered_df.groupby('Gender').agg({
                'Sleep Duration': ['mean', 'std'],
                'Quality of Sleep': ['mean', 'std'],
                'Stress Level': ['mean', 'std'],
                'Physical Activity Level': ['mean', 'std']
            }).round(2)
            st.dataframe(gender_metrics, use_container_width=True)
        
        with col2:
            # Sleep disorder by gender
            disorder_gender = pd.crosstab(filtered_df['Gender'], filtered_df['Sleep Disorder'])
            fig = px.bar(
                disorder_gender,
                title="Sleep Disorders by Gender",
                barmode='group',
                color_discrete_map={'Male': '#3498db', 'Female': '#e74c3c'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed comparison
        metrics = ['Sleep Duration', 'Quality of Sleep', 'Stress Level', 'Physical Activity Level']
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=metrics
        )
        
        for idx, metric in enumerate(metrics):
            row = idx // 2 + 1
            col = idx % 2 + 1
            
            for gender in filtered_df['Gender'].unique():
                data = filtered_df[filtered_df['Gender'] == gender][metric]
                fig.add_trace(
                    go.Box(y=data, name=gender, showlegend=(idx == 0)),
                    row=row, col=col
                )
        
        fig.update_layout(height=600, title_text="Gender Comparison Across Metrics")
        st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Age Group Analysis":
        st.subheader("Detailed Age Group Analysis")
        
        filtered_df['Age_Group'] = pd.cut(
            filtered_df['Age'],
            bins=[0, 30, 40, 50, 100],
            labels=['Under 30', '30-40', '40-50', '50+']
        )
        
        # Age group metrics
        age_metrics = filtered_df.groupby('Age_Group').agg({
            'Person ID': 'count',
            'Sleep Duration': 'mean',
            'Quality of Sleep': 'mean',
            'Stress Level': 'mean',
            'Physical Activity Level': 'mean',
            'Heart Rate': 'mean'
        }).round(2)
        age_metrics.columns = ['Count', 'Avg Sleep', 'Avg Quality', 'Avg Stress', 'Avg Activity', 'Avg Heart Rate']
        
        st.dataframe(age_metrics.style.background_gradient(cmap='coolwarm'), use_container_width=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.line(
                filtered_df.groupby('Age')['Sleep Duration'].mean().reset_index(),
                x='Age',
                y='Sleep Duration',
                title="Sleep Duration Trend by Age",
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            disorder_age = pd.crosstab(filtered_df['Age_Group'], filtered_df['Sleep Disorder'], normalize='index') * 100
            fig = px.bar(
                disorder_age,
                title="Sleep Disorder Distribution by Age Group (%)",
                barmode='stack'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "Sleep Disorder Analysis":
        st.subheader("Detailed Sleep Disorder Analysis")
        
        # Disorder comparison table
        disorder_metrics = filtered_df.groupby('Sleep Disorder').agg({
            'Person ID': 'count',
            'Age': 'mean',
            'Sleep Duration': 'mean',
            'Quality of Sleep': 'mean',
            'Stress Level': 'mean',
            'Physical Activity Level': 'mean',
            'Heart Rate': 'mean',
            'Systolic': 'mean',
            'Diastolic': 'mean'
        }).round(2)
        
        st.dataframe(disorder_metrics.style.background_gradient(cmap='RdYlGn_r'), use_container_width=True)
        
        # Risk factor analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # BMI distribution by disorder
            bmi_disorder = pd.crosstab(filtered_df['Sleep Disorder'], filtered_df['BMI Category'], normalize='index') * 100
            fig = px.bar(
                bmi_disorder,
                title="BMI Category Distribution by Sleep Disorder (%)",
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Stress level by disorder
            fig = px.violin(
                filtered_df,
                x='Sleep Disorder',
                y='Stress Level',
                color='Sleep Disorder',
                title="Stress Level Distribution by Sleep Disorder",
                box=True
            )
            st.plotly_chart(fig, use_container_width=True)

elif page == "📈 Statistical Analysis":
    st.title("📈 Statistical Analysis")
    
    st.markdown("### Hypothesis Testing & Statistical Insights")
    
    tab1, tab2, tab3 = st.tabs(["T-Tests", "ANOVA", "Chi-Square Tests"])
    
    with tab1:
        st.subheader("Independent T-Tests")
      #KPi color 
        st.markdown("""
        <style>
        /* ----- KPI BOX DESIGN ----- */
        [data-testid="stMetric"] {
            background-color: #202225;        /* رمادي غامق ناعم */
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 16px 12px;
            text-align: center;
            box-shadow: 0 1px 4px rgba(0,0,0,0.3);
        }

        /* العنوان (label) */
        [data-testid="stMetricLabel"] > div {
            color: rgba(255,255,255,0.85);
            font-weight: 600;
        }

        /* القيمة (الرقم الأساسي) */
        [data-testid="stMetricValue"] {
            color: #ffffff;
            font-size: 1.8rem;
            font-weight: 700;
        }

        /* الدلتا (↑ أو ↓) */
        [data-testid="stMetricDelta"] {
            font-weight: 600;
            border-radius: 20px;
            padding: 2px 8px;
            background-color: rgba(255,255,255,0.08);
        }

        /* تعديل اللون الأخضر والأحمر تلقائي */
        [data-testid="stMetricDelta"] svg {
            height: 1em;
        }

        /* إذا حابة تقللي الفرق بين الكروت */
        div[data-testid="stHorizontalBlock"] {
            gap: 0.8rem !important;
        }
        </style>
        """, unsafe_allow_html=True)

        # Gender comparison
        male_sleep = filtered_df[filtered_df['Gender'] == 'Male']['Sleep Duration'].dropna()
        female_sleep = filtered_df[filtered_df['Gender'] == 'Female']['Sleep Duration'].dropna()
        

        if len(male_sleep) > 0 and len(female_sleep) > 0:
            t_stat, p_value = stats.ttest_ind(male_sleep, female_sleep)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("T-Statistic", f"{t_stat:.4f}")
            with col2:
                st.metric("P-Value", f"{p_value:.4f}")
            with col3:
                result = "Significant" if p_value < 0.05 else "Not Significant"
                st.metric("Result (α=0.05)", result)
            
            st.info(f"""
            **Interpretation:** 
            {'There IS a statistically significant difference' if p_value < 0.05 else 'There is NO statistically significant difference'} 
            in sleep duration between males (μ={male_sleep.mean():.2f}) and females (μ={female_sleep.mean():.2f}).
            """)
        
        # Disorder vs No Disorder
        st.markdown("---")
        st.subheader("Sleep Disorder Impact on Quality")
        
        with_disorder = filtered_df[filtered_df['Sleep Disorder'] != 'None']['Quality of Sleep'].dropna()
        without_disorder = filtered_df[filtered_df['Sleep Disorder'] == 'None']['Quality of Sleep'].dropna()
        
        if len(with_disorder) > 0 and len(without_disorder) > 0:
            t_stat, p_value = stats.ttest_ind(without_disorder, with_disorder)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("T-Statistic", f"{t_stat:.4f}")
            with col2:
                st.metric("P-Value", f"{p_value:.4f}")
            with col3:
                result = "Significant" if p_value < 0.05 else "Not Significant"
                st.metric("Result (α=0.05)", result)
            
            # Visualization
            fig = go.Figure()
            fig.add_trace(go.Box(y=without_disorder, name='No Disorder', marker_color='#2ecc71'))
            fig.add_trace(go.Box(y=with_disorder, name='With Disorder', marker_color='#e74c3c'))
            fig.update_layout(title="Sleep Quality: Disorder vs No Disorder", yaxis_title="Quality Score")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("ANOVA - Analysis of Variance")
        
        # ANOVA for occupation groups
        st.markdown("#### Sleep Duration across Occupations")
        
        occupation_groups = [group['Sleep Duration'].dropna() for name, group in filtered_df.groupby('Occupation')]
        
        if len(occupation_groups) > 2:
            f_stat, p_value = stats.f_oneway(*occupation_groups)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("F-Statistic", f"{f_stat:.4f}")
            with col2:
                st.metric("P-Value", f"{p_value:.4f}")
            with col3:
                result = "Significant" if p_value < 0.05 else "Not Significant"
                st.metric("Result (α=0.05)", result)
            
            # Box plot
            fig = px.box(
                filtered_df,
                x='Occupation',
                y='Sleep Duration',
                color='Occupation',
                title="Sleep Duration Distribution by Occupation"
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        # ANOVA for BMI categories
        st.markdown("---")
        st.markdown("#### Sleep Quality across BMI Categories")
        
        bmi_groups = [group['Quality of Sleep'].dropna() for name, group in filtered_df.groupby('BMI Category')]
        
        if len(bmi_groups) > 2:
            f_stat, p_value = stats.f_oneway(*bmi_groups)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("F-Statistic", f"{f_stat:.4f}")
            with col2:
                st.metric("P-Value", f"{p_value:.4f}")
            with col3:
                result = "Significant" if p_value < 0.05 else "Not Significant"
                st.metric("Result (α=0.05)", result)
            
            fig = px.box(
                filtered_df,
                x='BMI Category',
                y='Quality of Sleep',
                color='BMI Category',
                title="Sleep Quality Distribution by BMI Category"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Chi-Square Tests of Independence")
        
        # Gender vs Sleep Disorder
        st.markdown("#### Gender vs Sleep Disorder")
        
        contingency_table = pd.crosstab(filtered_df['Gender'], filtered_df['Sleep Disorder'])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Chi-Square", f"{chi2:.4f}")
        with col2:
            st.metric("P-Value", f"{p_value:.4f}")
        with col3:
            st.metric("DOF", f"{dof}")
        with col4:
            result = "Dependent" if p_value < 0.05 else "Independent"
            st.metric("Result", result)
        
        # Display contingency table
        st.dataframe(contingency_table, use_container_width=True)
        
        # Heatmap
        fig = px.imshow(
            contingency_table,
            text_auto=True,
            aspect='auto',
            title="Contingency Table: Gender vs Sleep Disorder",
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # BMI vs Sleep Disorder
        st.markdown("---")
        st.markdown("#### BMI Category vs Sleep Disorder")
        
        contingency_table2 = pd.crosstab(filtered_df['BMI Category'], filtered_df['Sleep Disorder'])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table2)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Chi-Square", f"{chi2:.4f}")
        with col2:
            st.metric("P-Value", f"{p_value:.4f}")
        with col3:
            st.metric("DOF", f"{dof}")
        with col4:
            result = "Dependent" if p_value < 0.05 else "Independent"
            st.metric("Result", result)
        
        st.dataframe(contingency_table2, use_container_width=True)
        
        fig = px.imshow(
            contingency_table2,
            text_auto=True,
            aspect='auto',
            title="Contingency Table: BMI Category vs Sleep Disorder",
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "🎯 Predictive Insights":
    st.title("🎯 Predictive Insights & Risk Assessment")
    st.markdown("""
    <style>
    /* ----- KPI BOX DESIGN ----- */
    [data-testid="stMetric"] {
        background-color: #202225;        /* رمادي غامق ناعم */
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 16px 12px;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.3);
    }

    /* العنوان (label) */
    [data-testid="stMetricLabel"] > div {
        color: rgba(255,255,255,0.85);
        font-weight: 600;
    }

    /* القيمة (الرقم الأساسي) */
    [data-testid="stMetricValue"] {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 700;
    }

    /* الدلتا (↑ أو ↓) */
    [data-testid="stMetricDelta"] {
        font-weight: 600;
        border-radius: 20px;
        padding: 2px 8px;
        background-color: rgba(255,255,255,0.08);
    }

    /* تعديل اللون الأخضر والأحمر تلقائي */
    [data-testid="stMetricDelta"] svg {
        height: 1em;
    }

    /* إذا حابة تقللي الفرق بين الكروت */
    div[data-testid="stHorizontalBlock"] {
        gap: 0.8rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Risk Factors", "Sleep Quality Predictor", "Health Recommendations"])
    
    with tab1:
        st.subheader("Sleep Disorder Risk Factors")
        
        # Calculate risk scores
        risk_factors = filtered_df.copy()
        
        # Risk score calculation
        risk_factors['Risk_Score'] = 0
        risk_factors.loc[risk_factors['Sleep Duration'] < 6, 'Risk_Score'] += 2
        risk_factors.loc[risk_factors['Stress Level'] > 7, 'Risk_Score'] += 2
        risk_factors.loc[risk_factors['Physical Activity Level'] < 40, 'Risk_Score'] += 1
        risk_factors.loc[risk_factors['BMI Category'].isin(['Obese', 'Overweight']), 'Risk_Score'] += 1
        risk_factors.loc[risk_factors['Heart Rate'] > 80, 'Risk_Score'] += 1
        risk_factors.loc[risk_factors['Systolic'] > 130, 'Risk_Score'] += 1
        
        # Risk categories
        risk_factors['Risk_Category'] = pd.cut(
            risk_factors['Risk_Score'],
            bins=[-1, 2, 4, 10],
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            low_risk = (risk_factors['Risk_Category'] == 'Low Risk').sum()
            st.metric("🟢 Low Risk", f"{low_risk} ({low_risk/len(risk_factors)*100:.1f}%)")
        
        with col2:
            med_risk = (risk_factors['Risk_Category'] == 'Medium Risk').sum()
            st.metric("🟡 Medium Risk", f"{med_risk} ({med_risk/len(risk_factors)*100:.1f}%)")
        
        with col3:
            high_risk = (risk_factors['Risk_Category'] == 'High Risk').sum()
            st.metric("🔴 High Risk", f"{high_risk} ({high_risk/len(risk_factors)*100:.1f}%)")
        
        # Risk distribution
        col1, col2 = st.columns(2)
        
        with col1:
            risk_counts = risk_factors['Risk_Category'].value_counts()
            fig = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title="Risk Category Distribution",
                color_discrete_map={
                    'Low Risk': '#2ecc71',
                    'Medium Risk': '#f39c12',
                    'High Risk': '#e74c3c'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Risk by occupation
            risk_occ = pd.crosstab(risk_factors['Occupation'], risk_factors['Risk_Category'], normalize='index') * 100
            fig = px.bar(
                risk_occ,
                title="Risk Distribution by Occupation (%)",
                barmode='stack',
                color_discrete_map={
                    'Low Risk': '#2ecc71',
                    'Medium Risk': '#f39c12',
                    'High Risk': '#e74c3c'
                }
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance for sleep disorders
        st.markdown("### Key Risk Indicators")
        
        disorder_yes = filtered_df[filtered_df['Sleep Disorder'] != 'None']
        disorder_no = filtered_df[filtered_df['Sleep Disorder'] == 'None']
        
        factors = {
            'Low Sleep Duration (<6h)': [
                (disorder_yes['Sleep Duration'] < 6).mean() * 100,
                (disorder_no['Sleep Duration'] < 6).mean() * 100
            ],
            'High Stress (>7)': [
                (disorder_yes['Stress Level'] > 7).mean() * 100,
                (disorder_no['Stress Level'] > 7).mean() * 100
            ],
            'Low Activity (<40)': [
                (disorder_yes['Physical Activity Level'] < 40).mean() * 100,
                (disorder_no['Physical Activity Level'] < 40).mean() * 100
            ],
            'High BMI (Overweight/Obese)': [
                disorder_yes['BMI Category'].isin(['Obese', 'Overweight']).mean() * 100,
                disorder_no['BMI Category'].isin(['Obese', 'Overweight']).mean() * 100
            ],
            'Elevated Heart Rate (>80)': [
                (disorder_yes['Heart Rate'] > 80).mean() * 100,
                (disorder_no['Heart Rate'] > 80).mean() * 100
            ]
        }
        
        factors_df = pd.DataFrame(factors, index=['With Disorder', 'Without Disorder']).T
        
        fig = px.bar(
            factors_df,
            barmode='group',
            title="Prevalence of Risk Factors (%)",
            labels={'value': 'Percentage (%)', 'index': 'Risk Factor'},
            color_discrete_map={'With Disorder': '#e74c3c', 'Without Disorder': '#2ecc71'}
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Sleep Quality Predictor")
        st.markdown("*Interactive tool to predict sleep quality based on lifestyle factors*")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pred_age = st.slider("Age", 20, 70, 35)
            pred_sleep_duration = st.slider("Sleep Duration (hours)", 4.0, 10.0, 7.0, 0.1)
            pred_stress = st.slider("Stress Level (1-10)", 1, 10, 5)
            pred_activity = st.slider("Physical Activity (min/day)", 0, 120, 60)
        
        with col2:
            pred_steps = st.number_input("Daily Steps", 1000, 15000, 7000, 500)
            pred_bmi = st.selectbox("BMI Category", ['Normal', 'Normal Weight', 'Overweight', 'Obese'])
            pred_gender = st.radio("Gender", ['Male', 'Female'])
            pred_occupation = st.selectbox("Occupation", sorted(filtered_df['Occupation'].unique()))
        
        # Simple prediction model based on data patterns
        predicted_quality = 5.0  # Base score
        
        # Adjust based on factors
        if pred_sleep_duration >= 7 and pred_sleep_duration <= 9:
            predicted_quality += 2
        elif pred_sleep_duration < 6:
            predicted_quality -= 2
        
        if pred_stress <= 4:
            predicted_quality += 1.5
        elif pred_stress >= 8:
            predicted_quality -= 2
        
        if pred_activity >= 60:
            predicted_quality += 1
        elif pred_activity < 30:
            predicted_quality -= 1
        
        if pred_steps >= 7000:
            predicted_quality += 0.5
        elif pred_steps < 4000:
            predicted_quality -= 0.5
        
        if pred_bmi in ['Normal', 'Normal Weight']:
            predicted_quality += 0.5
        elif pred_bmi == 'Obese':
            predicted_quality -= 1
        
        # Cap between 1 and 10
        predicted_quality = max(1, min(10, predicted_quality))
        
        st.markdown("---")
        st.subheader("Predicted Sleep Quality")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=predicted_quality,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Sleep Quality Score"},
                delta={'reference': 7, 'increasing': {'color': "green"}},
                gauge={
                    'axis': {'range': [None, 10]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 4], 'color': "#ffcccc"},
                        {'range': [4, 7], 'color': "#ffffcc"},
                        {'range': [7, 10], 'color': "#ccffcc"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 7
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations based on prediction
        st.markdown("### Personalized Recommendations")
        
        recommendations = []
        
        if pred_sleep_duration < 7:
            recommendations.append("⏰ **Increase sleep duration**: Aim for 7-9 hours per night")
        
        if pred_stress > 7:
            recommendations.append("🧘 **Reduce stress**: Try meditation, yoga, or counseling")
        
        if pred_activity < 60:
            recommendations.append("🏃 **Increase physical activity**: Target 60+ minutes daily")
        
        if pred_steps < 7000:
            recommendations.append("👟 **Walk more**: Aim for 7,000-10,000 steps per day")
        
        if pred_bmi in ['Overweight', 'Obese']:
            recommendations.append("🥗 **Manage weight**: Consult a nutritionist for healthy BMI")
        
        if recommendations:
            for rec in recommendations:
                st.warning(rec)
        else:
            st.success("✅ Great! Your lifestyle factors are well-balanced for good sleep quality!")
    
    with tab3:
        st.subheader("Health Recommendations by Profile")
        
        profile_type = st.selectbox(
            "Select Profile:",
            ["All Profiles", "With Sleep Disorders", "High Stress", "Low Activity", "Poor Sleep Quality"]
        )
        
        if profile_type == "With Sleep Disorders":
            profile_data = filtered_df[filtered_df['Sleep Disorder'] != 'None']
            st.markdown("""
            ### Recommendations for People with Sleep Disorders
            
            #### Immediate Actions:
            - 🏥 **Consult a healthcare provider** for proper diagnosis and treatment
            - 📋 **Keep a sleep diary** tracking sleep patterns and triggers
            - 💊 **Follow prescribed treatments** consistently
            
            #### Lifestyle Modifications:
            - ⏰ Maintain consistent sleep-wake schedule
            - 🚫 Avoid caffeine 6 hours before bedtime
            - 📱 Reduce screen time 1 hour before sleep
            - 🏃 Regular exercise (but not close to bedtime)
            - 🍽️ Avoid heavy meals before sleep
            
            #### Sleep Environment:
            - 🌡️ Keep bedroom cool (60-67°F)
            - 🌑 Ensure complete darkness
            - 🔇 Minimize noise or use white noise
            - 🛏️ Invest in comfortable mattress and pillows
            """)
            
        elif profile_type == "High Stress":
            profile_data = filtered_df[filtered_df['Stress Level'] > 7]
            st.markdown("""
            ### Recommendations for High Stress Individuals
            
            #### Stress Management:
            - 🧘‍♀️ **Practice mindfulness meditation** (10-15 minutes daily)
            - 🗣️ **Talk therapy or counseling** for chronic stress
            - ✍️ **Journaling** to process thoughts and emotions
            - 🎵 **Relaxation techniques**: Deep breathing, progressive muscle relaxation
            
            #### Work-Life Balance:
            - ⏱️ Set boundaries between work and personal time
            - 🚫 Learn to say "no" to excessive commitments
            - 🎯 Prioritize tasks and delegate when possible
            - 📅 Schedule regular breaks throughout the day
            
            #### Physical Activity:
            - 🏃 Regular exercise reduces cortisol levels
            - 🌳 Outdoor activities for additional benefits
            - 🧘 Yoga combines physical activity with stress relief
            """)
            
        elif profile_type == "Low Activity":
            profile_data = filtered_df[filtered_df['Physical Activity Level'] < 40]
            st.markdown("""
            ### Recommendations for Low Activity Individuals
            
            #### Getting Started:
            - 👟 **Start small**: 10-minute walks, gradually increase
            - 🎯 **Set realistic goals**: Aim for 30 minutes, 5 days/week
            - 📱 **Use fitness apps** to track progress
            - 👥 **Find an exercise buddy** for motivation
            
            #### Activity Ideas:
            - 🚶 Walking or hiking
            - 🏊 Swimming (low-impact, full-body workout)
            - 🚴 Cycling or stationary bike
            - 💃 Dancing or group fitness classes
            - 🎾 Recreational sports
            
            #### Daily Habits:
            - 🪜 Take stairs instead of elevators
            - 🅿️ Park farther away from destinations
            - 📞 Take walking breaks during calls
            - 🏢 Stand or walk during TV commercials
            """)
            
        elif profile_type == "Poor Sleep Quality":
            profile_data = filtered_df[filtered_df['Quality of Sleep'] < 5]
            st.markdown("""
            ### Recommendations for Poor Sleep Quality
            
            #### Sleep Hygiene:
            - ⏰ **Consistent schedule**: Same bedtime and wake time daily
            - 🌅 **Morning sunlight exposure**: Regulates circadian rhythm
            - 🚫 **Limit naps**: Max 20-30 minutes, before 3 PM
            - ☕ **Cut caffeine**: No caffeine after 2 PM
            
            #### Bedroom Optimization:
            - 🌡️ Temperature: 60-67°F (15-19°C)
            - 🌑 Complete darkness (blackout curtains)
            - 🔇 Quiet environment (earplugs or white noise)
            - 🛏️ Comfortable bedding and supportive mattress
            
            #### Pre-Sleep Routine:
            - 📚 Relaxing activities: Reading, light stretching
            - 🛁 Warm bath 1-2 hours before bed
            - 📱 No screens 1 hour before sleep
            - 🧘 Relaxation exercises or meditation
            
            #### What to Avoid:
            - 🍷 Alcohol (disrupts sleep cycles)
            - 🍔 Heavy meals within 3 hours of bedtime
            - 💧 Too much liquid before bed
            - 🏋️ Vigorous exercise close to bedtime
            """)
        
        else:
            profile_data = filtered_df
            st.markdown("""
            ### General Health & Sleep Recommendations
            
            #### The 7 Pillars of Good Sleep:
            
            1. **⏰ Consistency**: Regular sleep-wake schedule
            2. **🏃 Activity**: 30+ minutes of moderate exercise daily
            3. **🧘 Stress Management**: Relaxation techniques
            4. **🥗 Nutrition**: Balanced diet, avoid late heavy meals
            5. **🌑 Environment**: Dark, quiet, cool bedroom
            6. **📱 Technology**: Limit screens before bed
            7. **☕ Substances**: Limit caffeine, alcohol, nicotine
            
            #### Health Metrics to Monitor:
            - 💤 Sleep duration: 7-9 hours
            - ❤️ Heart rate: 60-100 bpm at rest
            - 🩺 Blood pressure: <120/80 mmHg
            - ⚖️ BMI: 18.5-24.9 (Normal range)
            - 👟 Daily steps: 7,000-10,000
            - 🏃 Physical activity: 150+ minutes/week
            """)
        
        # Show profile statistics
        if len(profile_data) > 0:
            st.markdown("---")
            st.subheader("Profile Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("👥 Group Size", len(profile_data))
            with col2:
                st.metric("😴 Avg Sleep Duration", f"{profile_data['Sleep Duration'].mean():.1f} hrs")
            with col3:
                st.metric("⭐ Avg Sleep Quality", f"{profile_data['Quality of Sleep'].mean():.1f}/10")
            with col4:
                disorder_pct = (profile_data['Sleep Disorder'] != 'None').sum() / len(profile_data) * 100
                st.metric("⚠️ With Disorders", f"{disorder_pct:.1f}%")

elif page == "📋 Data Explorer":
    st.title("📋 Data Explorer")
    
    tab1, tab2, tab3 = st.tabs(["Raw Data", "Summary Statistics", "Custom Analysis"])
    
    with tab1:
        st.subheader("Raw Data View")
        
        # Search and filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            search_col = st.selectbox("Search in column:", filtered_df.columns)
        
        with col2:
            search_term = st.text_input("Search term:", "")
        
        with col3:
            show_rows = st.slider("Rows to display:", 10, 100, 25)
        
        # Apply search
        display_df = filtered_df.copy()
        if search_term:
            display_df = display_df[
                display_df[search_col].astype(str).str.contains(search_term, case=False, na=False)
            ]
        
        # Display data
        st.dataframe(
            display_df.head(show_rows).style.highlight_max(axis=0, subset=['Sleep Duration', 'Quality of Sleep']),
            use_container_width=True
        )
        
        # Data info
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Dataset Information")
            st.write(f"**Total rows:** {len(display_df)}")
            st.write(f"**Total columns:** {len(display_df.columns)}")
            st.write(f"**Memory usage:** {display_df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        
        with col2:
            st.markdown("### Missing Values")
            missing_data = display_df.isnull().sum()
            missing_data = missing_data[missing_data > 0]
            if len(missing_data) > 0:
                st.dataframe(missing_data, use_container_width=True)
            else:
                st.success("No missing values!")
    
    with tab2:
        st.subheader("Summary Statistics")
        
        # Numeric summary
        st.markdown("### Numeric Variables")
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
        st.dataframe(
            filtered_df[numeric_cols].describe().T.style.background_gradient(cmap='coolwarm'),
            use_container_width=True
        )
        
        # Categorical summary
        st.markdown("### Categorical Variables")
        categorical_cols = filtered_df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            with st.expander(f"📊 {col}"):
                value_counts = filtered_df[col].value_counts()
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.dataframe(value_counts, use_container_width=True)
                
                with col2:
                    fig = px.bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        title=f"Distribution of {col}",
                        labels={'x': col, 'y': 'Count'}
                    )
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Custom Analysis Builder")
        
        st.markdown("*Build your own visualizations*")
        
        chart_type = st.selectbox(
            "Select Chart Type:",
            ["Bar Chart", "Line Chart", "Scatter Plot", "Box Plot", "Histogram", "Pie Chart"]
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_axis = st.selectbox("X-axis:", filtered_df.columns, index=2)
        
        with col2:
            if chart_type not in ["Histogram", "Pie Chart"]:
                y_axis = st.selectbox("Y-axis:", filtered_df.columns, index=3)
        
        color_by = st.selectbox("Color by (optional):", ["None"] + list(filtered_df.columns))
        color_col = None if color_by == "None" else color_by
        
        # Generate chart
        st.markdown("---")
        
        try:
            if chart_type == "Bar Chart":
                if filtered_df[x_axis].dtype == 'object':
                    data = filtered_df.groupby(x_axis)[y_axis].mean().reset_index()
                    fig = px.bar(data, x=x_axis, y=y_axis, color=color_col, title=f"{y_axis} by {x_axis}")
                else:
                    fig = px.bar(filtered_df, x=x_axis, y=y_axis, color=color_col, title=f"{y_axis} vs {x_axis}")
                
            elif chart_type == "Line Chart":
                fig = px.line(filtered_df, x=x_axis, y=y_axis, color=color_col, title=f"{y_axis} over {x_axis}")
            
            elif chart_type == "Scatter Plot":
                fig = px.scatter(filtered_df, x=x_axis, y=y_axis, color=color_col, title=f"{y_axis} vs {x_axis}")
            
            elif chart_type == "Box Plot":
                fig = px.box(filtered_df, x=x_axis, y=y_axis, color=color_col, title=f"{y_axis} distribution by {x_axis}")
            
            elif chart_type == "Histogram":
                fig = px.histogram(filtered_df, x=x_axis, color=color_col, title=f"Distribution of {x_axis}")
            
            elif chart_type == "Pie Chart":
                value_counts = filtered_df[x_axis].value_counts()
                fig = px.pie(values=value_counts.values, names=value_counts.index, title=f"Distribution of {x_axis}")
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error creating chart: {str(e)}")
            st.info("Try selecting different columns or chart type.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d;'>
    <p>Sleep Health Analysis Dashboard | Built with Streamlit</p>
    <p>Data Source: Sleep Health and Lifestyle Dataset</p>
</div>
""", unsafe_allow_html=True)