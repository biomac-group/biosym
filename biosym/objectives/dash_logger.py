"""
Dash app for real-time visualization of objective function values during optimization.

This module provides a Dash/Plotly-based web interface to visualize how individual
objective terms evolve during IPOPT optimization.
"""

from dash import Dash, html, dcc, Input, Output
import plotly.graph_objects as go
import pandas as pd
from typing import Optional


def create_dashboard_app(iteration_logger=None, port: int = 8050):
    """
    Create a Dash app for visualizing objective function convergence.
    
    Parameters
    ----------
    iteration_logger : IterationLogger, optional
        The logger containing iteration data
    port : int, optional
        Port to run the Dash server on (default: 8050)
    
    Returns
    -------
    Dash
        Configured Dash application instance
    
    Examples
    --------
    >>> from biosym.objectives.dash_logger import create_dashboard_app
    >>> # After solving with logging enabled:
    >>> app = create_dashboard_app(problem.iteration_logger)
    >>> app.run(debug=False)
    """
    app = Dash(__name__)
    
    app.layout = html.Div([
        html.H1("Objectives Evaluation", style={'textAlign': 'center'}),
        
        dcc.Graph(id='objective-plot'),
        
        dcc.Interval(
            id='interval-component',
            interval=2000,  # Update every 2 seconds
            n_intervals=0
        ),
        
        html.Div(id='stats-div', style={'marginTop': 20, 'textAlign': 'center'})
    ], style={'backgroundColor': '#111111', 'color': '#FFFFFF'})
    
    @app.callback(
        [Output('objective-plot', 'figure'),
         Output('stats-div', 'children')],
        [Input('interval-component', 'n_intervals')]
    )
    def update_graph(n):
        """Update the plot with latest logged data."""
        if iteration_logger is None or not iteration_logger.log_data:
            # Return empty plot if no data
            fig = go.Figure()
            fig.update_layout(
                title="Waiting for solution...",
                xaxis_title="Iteration",
                yaxis_title="Weighted Objective Value",
                template="plotly_dark"
            )
            return fig, "No data yet"
        
        # Get data as DataFrame
        df = iteration_logger.get_dataframe()
        
        if df.empty:
            fig = go.Figure()
            fig.update_layout(
                title="No data logged yet",
                xaxis_title="Iteration",
                yaxis_title="Weighted Objective Value",
                template="plotly_dark"
            )
            return fig, "Empty log"
        
        # Create figure with traces for each objective
        fig = go.Figure()
        
        # Add a trace for each objective column (skip 'iteration')
        for col in df.columns:
            if col != 'iteration':
                fig.add_trace(go.Scatter(
                    x=df['iteration'],
                    y=df[col],
                    mode='lines+markers',
                    name=col,
                    line=dict(width=2),
                    marker=dict(size=8)
                ))
        
        fig.update_layout(
            xaxis_title="Iteration",
            yaxis_title="Weighted Objective Value (log scale)",
            yaxis_type="log",
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            ),
            height=600,
            template="plotly_dark"
        )
        
        # Create stats text
        # n_iterations = len(df)
        # last_iter = df['iteration'].iloc[-1] if n_iterations > 0 else 0
        stats_text = "I believe in you IPOPT!"
        
        return fig, stats_text
    
    return app


def visualize_convergence(iteration_logger, port: int = 8050, debug: bool = False):
    """
    Launch a Dash app to visualize objective convergence.
    
    This is a convenience function that creates and runs the dashboard.
    
    Parameters
    ----------
    iteration_logger : IterationLogger
        The logger containing iteration data
    port : int, optional
        Port to run the server on (default: 8050)
    debug : bool, optional
        Run in debug mode (default: False)
    
    Examples
    --------
    >>> from biosym.objectives.dash_logger import visualize_convergence
    >>> # After solving:
    >>> visualize_convergence(problem.iteration_logger)
    """
    app = create_dashboard_app(iteration_logger, port=port)
    print(f"Starting Dash server on http://localhost:{port}")
    print("Press Ctrl+C to stop")
    app.run(debug=debug, port=port)


if __name__ == '__main__':
    # Simple test with dummy data
    app = create_dashboard_app(None)
    app.run(debug=True)
