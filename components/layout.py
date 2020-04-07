import plotly.graph_objects as go


def create_empty_field(below=False):
    rad_circle = 9.15
    len_field = 105
    wid_field = 68

    circle_y = (wid_field / 2 - rad_circle) * 100 / wid_field
    circle_x = (len_field / 2 - rad_circle) * 100 / len_field

    y_box = ((wid_field - 40.32) / 2) * 100 / wid_field

    layout = go.Layout(
        plot_bgcolor="rgba(0,255,112,1)",
        xaxis=dict(range=[0, 100],
                   showgrid=False,
                   showticklabels=False),
        yaxis=dict(range=[0, 100],
                   showgrid=False,
                   showticklabels=False),
    )

    # Create traces
    fig = go.Figure(layout=layout)
    fig.add_trace(go.Scatter(
        showlegend=False,
        x=[50, 50],
        y=[100, 0],
        mode='lines',
        line=dict(color='white', width=2)))

    fig.add_trace(go.Scatter(
        showlegend=False,
        x=[0, (16 / len_field) * 100, (16 / len_field) * 100, 0],
        y=[100 - y_box, 100 - y_box, y_box, y_box],
        mode='lines',
        line=dict(color='white', width=2),

    ))

    fig.add_trace(go.Scatter(
        showlegend=False,
        x=[100, (1 - 16 / len_field) * 100, (1 - 16 / len_field) * 100, 100],
        y=[100 - y_box, 100 - y_box, y_box, y_box],
        mode='lines',
        line=dict(color='white', width=2),
    ))

    fig.add_trace(go.Scatter(
        showlegend=False,
        x=[11 / len_field * 100],
        y=[50],
        mode="markers",
        marker=dict(
            color='white',
            size=7
        ),
    ))

    fig.add_trace(go.Scatter(
        showlegend=False,
        x=[(1 - 11 / len_field) * 100],
        y=[50],
        mode="markers",
        marker=dict(
            color='white',
            size=7
        ),
    ))

    fig.add_trace(go.Scatter(
        showlegend=False,
        x=[50],
        y=[50],
        mode="markers",
        marker=dict(
            color='white',
            size=7
        ),
    ))

    # Add circles
    if below:
        fig.update_layout(
            shapes=[
                # unfilled circle
                dict(
                    type="circle",
                    xref="x",
                    yref="y",
                    x0=circle_x,
                    y0=circle_y,
                    x1=100 - circle_x,
                    y1=100 - circle_y,
                    line_color="white",
                    layer="below"
                ),
            ]
        )
    else:
        fig.update_layout(
            shapes=[
                # unfilled circle
                dict(
                    type="circle",
                    xref="x",
                    yref="y",
                    x0=circle_x,
                    y0=circle_y,
                    x1=100 - circle_x,
                    y1=100 - circle_y,
                    line_color="white"
                ),
            ]
        )

    fig.update_layout(
        autosize=False,
        width=105 * 8,
        height=68 * 8)

    return fig
