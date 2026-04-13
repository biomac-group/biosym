"""
Dash app for real-time visualization of objective function values during optimization.

This module provides a Dash/Plotly-based web interface to visualize how individual
objective terms evolve during IPOPT optimization.
"""

from dash import Dash, html, dcc, Input, Output, State, callback_context, no_update
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Optional
from biosym.visualization import stickfigure


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
        html.H1("Optimization Progress", style={'textAlign': 'center'}),
        
        html.Div([
            html.Label("Y-Axis Scale:", style={'marginRight': '10px'}),
            dcc.RadioItems(
                id='scale-selector',
                options=[
                    {'label': 'Log Scale', 'value': 'log'},
                    {'label': 'Linear Scale', 'value': 'linear'}
                ],
                value='log',
                inline=True,
                inputStyle={"margin-right": "5px", "margin-left": "10px"}
            )
        ], style={'textAlign': 'center', 'marginBottom': '20px'}),

        html.Div([
            html.Div([
                html.H3("Objectives", style={'textAlign': 'center'}),
                dcc.Graph(id='objective-plot', style={'height': '600px'}),
            ], style={'width': '49%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            
            html.Div([
                html.H3("Constraints", style={'textAlign': 'center'}),
                dcc.Graph(id='constraint-plot', style={'height': '600px'}),
            ], style={'width': '49%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        ], style={'display': 'flex', 'justifyContent': 'space-between'}),
        
        dcc.Interval(
            id='interval-component',
            interval=500,  # Poll twice per second for smoother streaming
            n_intervals=0
        ),
        
        html.Div(id='stats-div', style={'marginTop': 20, 'textAlign': 'center'}),
        dcc.Store(id='last-update-iteration', data={'iteration': 0, 'rows': 0})
    ], style={'backgroundColor': '#111111', 'color': '#FFFFFF'})

    @app.callback(
        [Output('objective-plot', 'figure'),
         Output('constraint-plot', 'figure'),
         Output('stats-div', 'children'),
         Output('last-update-iteration', 'data')],
        [Input('interval-component', 'n_intervals'),
         Input('scale-selector', 'value')],
        [State('last-update-iteration', 'data')]
    )
    def update_graph(n, scale_type, last_update_state):
        """
        Update the plot with latest logged data.
        
        This callback is triggered every 2 seconds by the interval, but it will
        only redraw the plots if the iteration count has increased by at least
        100 since the last update, or if the scale type has changed.
        """
        ctx = callback_context
        
        if not ctx.triggered:
            return no_update

        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if iteration_logger is None or not iteration_logger.log_data:
            # Return empty plots if no data
            empty_fig = go.Figure()
            empty_fig.update_layout(
                title="Waiting for solution...",
                xaxis_title="Iteration",
                yaxis_title=f"Value ({scale_type} scale)",
                yaxis_type=scale_type,
                template="plotly_dark"
            )
            return empty_fig, empty_fig, "No data yet", no_update
        
        # Get data as DataFrame
        df = iteration_logger.get_dataframe()
        
        if df.empty:
            empty_fig = go.Figure()
            empty_fig.update_layout(
                title="No data logged yet",
                xaxis_title="Iteration",
                yaxis_title=f"Value ({scale_type} scale)",
                yaxis_type=scale_type,
                template="plotly_dark"
            )
            return empty_fig, empty_fig, "Empty log", no_update
            
        current_iter = int(df['iteration'].iloc[-1])
        current_rows = len(df)

        if isinstance(last_update_state, dict):
            last_update_iter = int(last_update_state.get('iteration', 0))
            last_update_rows = int(last_update_state.get('rows', 0))
        elif isinstance(last_update_state, (int, float)):
            last_update_iter = int(last_update_state)
            last_update_rows = 0
        else:
            last_update_iter = 0
            last_update_rows = 0
        
        # Determine if we should update
        should_update = False
        new_last_state = no_update
        
        if trigger_id != 'interval-component':
            # Always update on manual interaction (scale change)
            should_update = True
            new_last_state = {'iteration': current_iter, 'rows': current_rows}
        else:
            # Interval trigger: redraw whenever the logged snapshot changed.
            if current_iter < last_update_iter or current_rows < last_update_rows:
                should_update = True
                new_last_state = {'iteration': current_iter, 'rows': current_rows}
            elif current_iter != last_update_iter or current_rows != last_update_rows:
                should_update = True
                new_last_state = {'iteration': current_iter, 'rows': current_rows}
        
        if not should_update:
            return no_update
        
        # Separate objective and constraint columns
        all_cols = [col for col in df.columns if col != 'iteration']
        constraint_cols = [col for col in all_cols if col.startswith('Constraint_')]
        objective_cols = [col for col in all_cols if not col.startswith('Constraint_') and col != 'Total']
        
        # --- Objectives Plot ---
        fig_obj = go.Figure()
        
        # Calculate total objective
        if objective_cols:
            df['Total_Obj'] = df[objective_cols].sum(axis=1)
            
            # Add a trace for each objective column
            for col in objective_cols:
                fig_obj.add_trace(go.Scatter(
                    x=df['iteration'],
                    y=df[col],
                    mode='lines+markers',
                    name=col,
                    line=dict(width=2),
                    marker=dict(size=6)
                ))
            
            # Add Total Objective trace
            fig_obj.add_trace(go.Scatter(
                x=df['iteration'],
                y=df['Total_Obj'],
                mode='lines+markers',
                name='Total Objective',
                line=dict(width=4, color='white'),
                marker=dict(size=8, color='white')
            ))
        
        fig_obj.update_layout(
            xaxis_title="Iteration",
            yaxis_title=f"Weighted Objective Value",
            yaxis_type=scale_type,
            hovermode='x unified',
            uirevision='constant',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(0,0,0,0.5)"
            ),
            height=600,
            template="plotly_dark",
            margin=dict(l=40, r=40, t=40, b=40)
        )

        # --- Constraints Plot ---
        fig_cons = go.Figure()
        
        if constraint_cols:
            # Calculate total constraint violation
            df['Total_Cons'] = df[constraint_cols].sum(axis=1)
            
            # Add a trace for each constraint column
            for col in constraint_cols:
                # Remove "Constraint_" prefix for cleaner legend
                display_name = col.replace("Constraint_", "")
                fig_cons.add_trace(go.Scatter(
                    x=df['iteration'],
                    y=df[col],
                    mode='lines+markers',
                    name=display_name,
                    line=dict(width=2),
                    marker=dict(size=6)
                ))
                
            # Add Total Constraint Violation trace
            fig_cons.add_trace(go.Scatter(
                x=df['iteration'],
                y=df['Total_Cons'],
                mode='lines+markers',
                name='Total Violation',
                line=dict(width=4, color='white'),
                marker=dict(size=8, color='white')
            ))
            
        fig_cons.update_layout(
            xaxis_title="Iteration",
            yaxis_title=f"Weighted Constraint Violation (L1)",
            yaxis_type=scale_type,
            hovermode='x unified',
            uirevision='constant',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(0,0,0,0.5)"
            ),
            height=600,
            template="plotly_dark",
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        stats_text = f"Iteration: {df['iteration'].iloc[-1]}"
        
        # Return the new iteration count to store it
        return fig_obj, fig_cons, stats_text, new_last_state
    
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
