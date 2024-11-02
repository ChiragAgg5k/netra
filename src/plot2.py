import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict

def create_bubble_chart(df, filename):
    """
    Create an interactive bubble chart showing relationship between categories and sub-categories.
    Size of bubbles represents number of cases.
    """
    # Group by category and sub_category to get counts
    grouped_data = df.groupby(['category', 'sub_category']).size().reset_index(name='count')
    
    # Calculate category totals for color grouping
    category_totals = df.groupby('category').size()
    
    fig = px.scatter(grouped_data,
                    x='category',
                    y='sub_category',
                    size='count',
                    color='category',
                    hover_data=['count'],
                    title='Cybercrime Categories and Sub-categories Distribution',
                    width=1200,
                    height=800)
    
    fig.update_layout(
        xaxis_title="Main Category",
        yaxis_title="Sub Category",
        showlegend=True,
        xaxis={'tickangle': 45},
        title_x=0.5,
        title_font_size=20,
    )
    
    fig.write_html(filename)

def create_sunburst_chart(df, filename):
    """
    Create an interactive sunburst chart showing hierarchical relationship
    between categories, sub-categories, and crime details.
    """
    # Create hierarchical structure
    grouped_data = df.groupby(['category', 'sub_category', 'crimeaditionalinfo']).size().reset_index(name='count')
    
    fig = px.sunburst(grouped_data,
                      path=['category', 'sub_category', 'crimeaditionalinfo'],
                      values='count',
                      title='Hierarchical View of Cybercrime Categories',
                      width=1000,
                      height=1000)
    
    fig.update_layout(
        title_x=0.5,
        title_font_size=20,
    )
    
    fig.write_html(filename)

def main():
    # Read the datasets
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    
    # Combine datasets for visualization
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Create visualizations
    create_bubble_chart(combined_df, 'cybercrime_bubble_chart.html')
    create_sunburst_chart(combined_df, 'cybercrime_sunburst.html')

if __name__ == "__main__":
    main()