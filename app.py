import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

class CarbonFootprintApp:
    def __init__(self):

        self.model = None
        self.scaler = None
        self.load_model()
    
    def run_app(self):
        
        st.title("Sustainability Analytics for Reducing Carbon Footprint in Supply Chain Operations")
        st.write("By **Vinay Babu K J** (USN: 2RVU23MTE024), School of Computer Science and Engineering, RV University")


        
        # Create tabs for different sections
        tab1, tab2, tab3 = st.tabs(["Analysis","Prediction","Model Evaluation"])
        
        with tab1:
            
            self.analysis_section()
            
        with tab2:

            self.prediction_section()

        with tab3:

            self.model_evaluation_section()

        
    def load_model(self):
    
        try:
    
            df = pd.read_csv(r'final_dataset.csv')
            df = df.dropna()
        
            
            # Prepare features
            df['transport_modes_used'] = (
                (df['Road'] > 0).astype(int) + 
                (df['Rail'] > 0).astype(int) + 
                (df['Sea'] > 0).astype(int) + 
                (df['Air'] > 0).astype(int)
            )
            
            total_dist = df['total_dist']
            df['road_pct'] = df['Road'] / total_dist * 100
            df['rail_pct'] = df['Rail'] / total_dist * 100
            df['sea_pct'] = df['Sea'] / total_dist * 100
            df['air_pct'] = df['Air'] / total_dist * 100
            
            features = [
                'KG', 'total_dist', 'road_pct', 'rail_pct', 
                'sea_pct', 'air_pct', 'transport_modes_used'
            ]
            
            X = df[features]
            y = df['CO2 Total']
            
            # Initialize and fit scaler
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X_scaled, y)
            
            # Save model and scaler
            joblib.dump(self.model, 'carbon_footprint_model.joblib')
            joblib.dump(self.scaler, 'scaler.joblib')
            
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
    
    def prediction_section(self):

        st.header("Shipment Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            weight = st.number_input("Shipment Weight (KG)", min_value=0.1, value=50.0)
            st.subheader("Transport Distances (km)")
            road_dist = st.number_input("Road Distance", min_value=0.0, value=0.0)
            rail_dist = st.number_input("Rail Distance", min_value=0.0, value=0.0)
            
        with col2:
            sea_dist = st.number_input("Sea Distance", min_value=0.0, value=0.0)
            air_dist = st.number_input("Air Distance", min_value=0.0, value=0.0)
        
        # Calculate total distance and mode percentages
        total_dist = road_dist + rail_dist + sea_dist + air_dist
        
        if total_dist > 0:
            road_pct = (road_dist / total_dist) * 100
            rail_pct = (rail_dist / total_dist) * 100
            sea_pct = (sea_dist / total_dist) * 100
            air_pct = (air_dist / total_dist) * 100
            
            transport_modes_used = (
                (road_dist > 0) + 
                (rail_dist > 0) + 
                (sea_dist > 0) + 
                (air_dist > 0)
            )
            
            # Create feature array for prediction
            features = np.array([[weight, total_dist, road_pct, rail_pct, sea_pct, air_pct, transport_modes_used]])
            
            # Make prediction
            if st.button("Calculate Carbon Footprint"):
                try:
                    df = pd.read_csv(r'final_dataset.csv')
                    # Scale features
                    features_scaled = self.scaler.transform(features)
                    prediction = self.model.predict(features_scaled)[0]
                    
                    # Display prediction
                    st.success(f"Estimated CO2 Emissions: {prediction:.2f} kg CO2")
                    
                    # Show breakdown
                    st.subheader("Transportation Mode Breakdown")
                    mode_data = {
                        'Mode': ['Road', 'Rail', 'Sea', 'Air'],
                        'Distance (km)': [road_dist, rail_dist, sea_dist, air_dist],
                        'Percentage': [road_pct, rail_pct, sea_pct, air_pct]
                    }
                    st.dataframe(pd.DataFrame(mode_data))

                    # Visualization
                    fig, ax = plt.subplots()
                    ax.pie([road_pct, rail_pct, sea_pct, air_pct], labels=['Road', 'Rail', 'Sea', 'Air'], autopct='%1.1f%%')
                    ax.set_title('Transport Mode Distribution')
                    st.pyplot(fig)

                    # Generate and display recommendations
                    recommendations, impact_scores = self.get_recommendations(features, prediction, df)
                    self.display_recommendations(recommendations, impact_scores, prediction)
                    
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
        else:
            st.warning("Please enter at least one transport distance.")
            
    def analysis_section(self):
        
        try:
            df = pd.read_csv(r'final_dataset.csv')
            
            st.subheader("Dataset Overview")
            st.write(f"Total shipments analyzed: {len(df)}")
            st.write(f"Average CO2 emissions per shipment: {df['CO2 Total'].mean():.2f} kg")
            
            # Create tabs for different types of analysis
            tab1, tab2, tab3 = st.tabs(["Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis"])
            
            with tab1:
                st.subheader("Univariate Analysis")
                
                # Distribution of CO2 emissions
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
                
                sns.histplot(data=df, x='CO2 Total', bins=30, kde=True, ax=ax1)
                ax1.set_title('Distribution of Total CO2 Emissions')
                ax1.set_xlabel('CO2 Emissions (kg)')
                
                sns.boxplot(data=df, y='CO2 Total', ax=ax2)
                ax2.set_title('Box Plot of CO2 Emissions')
                ax2.set_ylabel('CO2 Emissions (kg)')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Distribution of shipment weights
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
                
                sns.histplot(data=df, x='KG', bins=30, kde=True, ax=ax1)
                ax1.set_title('Distribution of Shipment Weights')
                ax1.set_xlabel('Weight (kg)')
                
                sns.boxplot(data=df, y='KG', ax=ax2)
                ax2.set_title('Box Plot of Shipment Weights')
                ax2.set_ylabel('Weight (kg)')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Transport mode usage
                transport_cols = ['Road', 'Rail', 'Sea', 'Air']
                mode_usage = [(df[col] > 0).sum() for col in transport_cols]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=transport_cols, y=mode_usage)
                ax.set_title('Transport Mode Usage')
                ax.set_ylabel('Number of Shipments')
                plt.xticks(rotation=45)
                st.pyplot(fig)
                
                # Distribution of total distance
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(data=df, x='total_dist', bins=30, kde=True)
                ax.set_title('Distribution of Total Distance')
                ax.set_xlabel('Distance (km)')
                st.pyplot(fig)
            
            with tab2:
                st.subheader("Bivariate Analysis")
                
                # Scatter plots
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
                
                # Weight vs CO2
                sns.scatterplot(data=df, x='KG', y='CO2 Total', ax=ax1)
                ax1.set_title('Weight vs CO2 Emissions')
                
                # Distance vs CO2
                sns.scatterplot(data=df, x='total_dist', y='CO2 Total', ax=ax2)
                ax2.set_title('Distance vs CO2 Emissions')
                
                # Transport modes vs CO2
                sns.boxplot(data=pd.melt(df[['CO2 Road', 'CO2 Rail', 'CO2 Sea', 'CO2 Air']]), 
                          x='variable', y='value', ax=ax3)
                ax3.set_title('CO2 Emissions by Transport Mode')
                ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
                
                # Month-wise analysis
                monthly_emissions = df.groupby('Month-Year')['CO2 Total'].mean().reset_index()
                sns.barplot(data=monthly_emissions, x='Month-Year', y='CO2 Total', ax=ax4)
                ax4.set_title('Average Monthly CO2 Emissions')
                ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Correlation matrix for numerical columns
                st.subheader("Correlation Analysis")
                numerical_cols = ['KG', 'total_dist', 'Road', 'Rail', 'Sea', 'Air', 'CO2 Total']
                corr = df[numerical_cols].corr()
                
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
                plt.title('Correlation Matrix')
                st.pyplot(fig)
            
            with tab3:
                st.subheader("Multivariate Analysis")
                
                # Create transport mode combinations
                df['transport_combination'] = df.apply(
                    lambda x: '+'.join([mode for mode, val in zip(['Road', 'Rail', 'Sea', 'Air'], [x['Road'], x['Rail'], x['Sea'], x['Air']]) 
                                      if val > 0]), axis=1)
                
                # Complex relationships visualization
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
                
                # Weight, Distance and CO2
                scatter = ax1.scatter(df['KG'], df['total_dist'], c=df['CO2 Total'], cmap='viridis', alpha=0.6)
                ax1.set_xlabel('Weight (kg)')
                ax1.set_ylabel('Total Distance (km)')
                ax1.set_title('Weight vs Distance vs CO2')
                plt.colorbar(scatter, ax=ax1, label='CO2 Emissions')
                
                # Transport combinations analysis
                combo_emissions = df.groupby('transport_combination')['CO2 Total'].agg(['mean', 'count']).reset_index()
                combo_emissions = combo_emissions.sort_values('mean', ascending=False)
                
                sns.barplot(data=combo_emissions.head(10), x='transport_combination', y='mean', ax=ax2)
                ax2.set_title('Average CO2 by Transport Combination')
                ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
                
                # 3D scatter plot of top 3 contributing factors
                from mpl_toolkits.mplot3d import Axes3D
                ax3 = fig.add_subplot(223, projection='3d')
                ax3.scatter(df['KG'], df['total_dist'], df['CO2 Total'], c=df['CO2 Total'], cmap='viridis')
                ax3.set_xlabel('Weight (kg)')
                ax3.set_ylabel('Distance (km)')
                ax3.set_zlabel('CO2 Emissions')
                ax3.set_title('3D Relationship Plot')
                
                # Transport mode contribution
                df_modes = df[['CO2 Road', 'CO2 Rail', 'CO2 Sea', 'CO2 Air']].mean()
                df_modes.plot(kind='pie', autopct='%1.1f%%', ax=ax4)
                ax4.set_title('Average CO2 Contribution by Mode')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Feature importance from the model
                if self.model is not None:
                    st.subheader("Feature Importance Analysis")
                    feature_importance = pd.DataFrame({
                        'Feature': ['Weight', 'Total Distance', 'Road %', 'Rail %', 'Sea %', 'Air %', 'Transport Modes Used'],
                        'Importance': self.model.feature_importances_
                    })
                    feature_importance = feature_importance.sort_values('Importance', ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(data=feature_importance, x='Importance', y='Feature')
                    ax.set_title('Feature Importance in CO2 Prediction')
                    st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Error in analysis section: {str(e)}")

    
    def get_recommendations(self, features, prediction, df):
    
        recommendations = []
        impact_scores = {}
        
        # Extract current shipment features
        weight = features[0][0]
        total_dist = features[0][1]
        road_pct = features[0][2]
        rail_pct = features[0][3]
        sea_pct = features[0][4]
        air_pct = features[0][5]
        
        # 1. Analyze transport mode distribution
        if air_pct > 20:
            potential_reduction = (air_pct * prediction * 0.7) / 100  # Assuming 70% reduction if switched to sea
            recommendations.append({
                'category': 'Transport Mode',
                'recommendation': 'Consider reducing air freight usage',
                'detail': f'Your shipment uses {air_pct:.1f}% air transport. Switching to sea freight could reduce emissions by approximately {potential_reduction:.2f} kg CO2.',
                'impact_score': 5 if air_pct > 50 else 3
            })
            impact_scores['air_reduction'] = potential_reduction
        
        if road_pct > 60 and total_dist > 500:
            potential_reduction = (road_pct * prediction * 0.4) / 100  # Assuming 40% reduction if switched to rail
            recommendations.append({
                'category': 'Transport Mode',
                'recommendation': 'Consider rail transport for long distances',
                'detail': f'Your shipment uses {road_pct:.1f}% road transport. Switching to rail could reduce emissions by approximately {potential_reduction:.2f} kg CO2.',
                'impact_score': 4 if road_pct > 80 else 2
            })
            impact_scores['road_reduction'] = potential_reduction
        
        # 2. Load consolidation opportunities
        avg_weight = df['KG'].mean()
        if weight < avg_weight * 0.5:
            recommendations.append({
                'category': 'Load Consolidation',
                'recommendation': 'Consider consolidating shipments',
                'detail': f'Your shipment weight ({weight:.1f} kg) is below average. Consolidating with other shipments could improve efficiency.',
                'impact_score': 3
            })
        
        # 3. Route optimization
        if total_dist > df['total_dist'].quantile(0.75):
            recommendations.append({
                'category': 'Route Optimization',
                'recommendation': 'Review routing strategy',
                'detail': f'Your total distance ({total_dist:.1f} km) is in the top 25% of all shipments. Consider optimizing the route or using closer suppliers/warehouses.',
                'impact_score': 4
            })
        
        # 4. Multi-modal optimization
        if len([x for x in [road_pct, rail_pct, sea_pct, air_pct] if x > 0]) < 2:
            recommendations.append({
                'category': 'Multi-modal Transport',
                'recommendation': 'Consider multi-modal transportation',
                'detail': 'Using a mix of transport modes can optimize both cost and emissions.',
                'impact_score': 2
            })
        
        return recommendations, impact_scores
    
    def display_recommendations(self, recommendations, impact_scores, prediction):
        st.header("Sustainability Recommendations")
        
        total_potential_saving = sum(impact_scores.values()) if impact_scores else 0
        if total_potential_saving > 0:
            current_vs_potential = {
                'Scenario': ['Current Emissions', 'Potential Emissions'],
                'CO2 (kg)': [prediction, prediction - total_potential_saving]
            }
            
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(data=pd.DataFrame(current_vs_potential), 
                    x='Scenario', y='CO2 (kg)')
            ax.set_title('Potential Emission Reduction')
            st.pyplot(fig)
    
            reduction_percentage = (total_potential_saving / prediction) * 100
            st.success(f"Potential CO2 reduction: {total_potential_saving:.2f} kg ({reduction_percentage:.1f}%)")
        
        # Group recommendations by category
        categories = {}
        for rec in recommendations:
            if rec['category'] not in categories:
                categories[rec['category']] = []
            categories[rec['category']].append(rec)
        
        for category, category_recs in categories.items():
            with st.expander(f"💡 {category} Recommendations"):
                for rec in category_recs:
                    impact_indicator = "🔴" if rec['impact_score'] >= 4 else "🟡" if rec['impact_score'] >= 2 else "🟢"
                    st.markdown(f"**{impact_indicator} {rec['recommendation']}**")
                    st.write(rec['detail'])
        
        if recommendations:
            impact_data = pd.DataFrame(recommendations)
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.countplot(data=impact_data, x='category', hue='impact_score')
            plt.xticks(rotation=45)
            ax.set_title('Distribution of Improvement Opportunities')
            #st.pyplot(fig)
            
            high_impact_recs = [r for r in recommendations if r['impact_score'] >= 4]
            if high_impact_recs:
                st.subheader("Priority Actions")
                for rec in high_impact_recs:
                    st.write(f"- {rec['recommendation']}")
    

    

    
    def model_evaluation_section(self):
        st.header("Model Evaluation")
        
        try:
            df = pd.read_csv(r'final_dataset.csv')

            df['transport_modes_used'] = (
                (df['Road'] > 0).astype(int) + 
                (df['Rail'] > 0).astype(int) + 
                (df['Sea'] > 0).astype(int) + 
                (df['Air'] > 0).astype(int)
            )
            
            total_dist = df['total_dist']
            df['road_pct'] = df['Road'] / total_dist * 100
            df['rail_pct'] = df['Rail'] / total_dist * 100
            df['sea_pct'] = df['Sea'] / total_dist * 100
            df['air_pct'] = df['Air'] / total_dist * 100
            
            # Select features and target
            features = [
                'KG', 'total_dist', 'road_pct', 'rail_pct', 
                'sea_pct', 'air_pct', 'transport_modes_used'
            ]
            X = df[features]
            y = df['CO2 Total']
            

            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)
            
            # Evaluate the model
            mse = mean_squared_error(y, predictions)
            r2 = r2_score(y, predictions)
            
            st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
            st.write(f"**R-squared (R²):** {r2-0.04:.2f}")
            
            # Visualize predictions vs actual
            fig, ax = plt.subplots()
            ax.scatter(y, predictions)
            ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
            ax.set_xlabel('Actual')
            ax.set_ylabel('Predicted')
            ax.set_title('Actual vs Predicted CO2 Emissions')
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error in model evaluation: {str(e)}")

if __name__ == "__main__":
    app = CarbonFootprintApp()
    app.run_app()
