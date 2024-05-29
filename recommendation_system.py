
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import torch
import pickle
import requests
from flask import Flask
from sklearn.preprocessing import LabelEncoder
import scipy.sparse as ssp

# Load the datasets
user_movies = pd.read_csv(r"C:\Users\AHMED OSAMA\Desktop\code\recommender\final project\final_dash\recom\user_movies.csv")
rating_history_norm = pd.read_csv(r"C:\Users\AHMED OSAMA\Desktop\code\recommender\final project\final_dash\recom\rating_history_norm.csv", index_col=0)
with open(r'C:\Users\AHMED OSAMA\Desktop\code\recommender\final project\final_dash\recom\one_hot_encoder_sec.pkl', 'rb') as f:
    active_ohe = pickle.load(f)

movies_df = pd.read_csv(r'C:\Users\AHMED OSAMA\Desktop\code\recommender\final project\final_dash\recom\movies.csv')
links_df = pd.read_csv(r'C:\Users\AHMED OSAMA\Desktop\code\recommender\final project\final_dash\recom\links.csv')
loaded_sparse_matrix = ssp.load_npz(r'C:\Users\AHMED OSAMA\Desktop\code\recommender\final project\final_dash\recom\sparse_similarity_matrix.npz')
similarity = pd.DataFrame.sparse.from_spmatrix(loaded_sparse_matrix)
movie_encoder = LabelEncoder()
movies_df['movieId_encoded'] = movie_encoder.fit_transform(movies_df['movieId'])

# Factorization machines class
class FactorizationMachine(torch.nn.Module):
    def __init__(self, n, k, bias=False):
        super(FactorizationMachine, self).__init__()
        self.n = n
        self.k = k
        self.linear = torch.nn.Linear(self.n, 1, bias)
        self.V = torch.nn.Parameter(torch.randn(n, k))  
    def forward(self, x_batch):
        part_1 = torch.sum(torch.mm(x_batch, self.V).pow(2), dim=1, keepdim=True)
        part_2 = torch.sum(torch.mm(x_batch.pow(2), self.V.pow(2)), dim=1, keepdim=True)
        inter_term = 0.5 * (part_1 - part_2)
        var_strength = self.linear(x_batch)
        return var_strength + inter_term
    

# initialize the model
model=FactorizationMachine(n=29721, k=20)
model.load_state_dict(torch.load(r'C:\Users\AHMED OSAMA\Desktop\code\recommender\final project\final_dash\recom\model_cola.pt'))

# Define the recommendation function
def recommend_movie(USER_ID,NUMBER_RECO):
    last_movie_seen=user_movies[user_movies.userId==USER_ID].iloc[-1]['imdbId']
    user_movies.loc[:,"last_seen"]=last_movie_seen

    #form user_can_rate dataframe
    user_rated = user_movies[user_movies['userId']==USER_ID]
    user_can_rate = user_movies[ ~user_movies.imdbId.isin (user_rated["imdbId"])]
    user_can_rate.loc[:,"userId"]=USER_ID
    user_can_rate= user_can_rate.drop_duplicates().reset_index(drop=True)
    user_can_rate = user_can_rate.sort_values(by='average_rating',ascending=False).head(2000)


    # form the features datafram
    cat_cols__ = user_can_rate.drop(columns=['userId', 'imdbId', 'last_seen' ,'rating','average_rating'])
    agg_history__ = user_can_rate[['userId']].merge(rating_history_norm, left_on='userId', right_index=True) 
    active_groups__ = active_ohe.transform(user_can_rate[['userId','imdbId','last_seen']]) 
    features = np.hstack((active_groups__, agg_history__.iloc[:,1:], cat_cols__))

    # predcit using the model
    features_tensor = torch.tensor(features, dtype=torch.float32)
    with torch.no_grad():
        model.eval()
        y = model(features_tensor)

    # form a sorted list of the top n movies
    ratingss=y.numpy().round(2).reshape(-1,1)
    movies=user_can_rate['imdbId'].values.reshape(-1,1)
    result = np.concatenate((ratingss, movies), axis=1)

    top_watched = user_rated.sort_values(by='rating', ascending=False)["imdbId"].head(10).tolist()

    return result[np.argsort(result[:, 0][::-1])][:NUMBER_RECO,1].astype(int) ,  top_watched

# Define the similar movies function
def recommend_movies(movie_name):
    try:
        movie_index = movies_df[movies_df['title'] == movie_name].index[0]
        movie_similarity = similarity[movie_index]
        similar_indices = movie_similarity.argsort()[-11:][::-1]  # Get top 10 similar movies
        similar_movie_ids = movies_df.iloc[similar_indices]['movieId'].tolist()
        return similar_movie_ids
    except IndexError:
        return f"Movie '{movie_name}' not found in the dataset."
    except Exception as e:
        return str(e)

# Fetch movie details using the OMDb API
def fetch_movie_details(imdb_id):
    if imdb_id:
        imdb_id_str = f"tt{str(imdb_id).zfill(7)}"
        api_key = 'd0b8676c'
        url = f'http://www.omdbapi.com/?i={imdb_id_str}&apikey={api_key}'
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data['Response'] == 'True':
                title = data.get('Title', 'N/A')
                release_date = data.get('Released', 'N/A')
                overview = data.get('Plot', 'N/A')
                poster_path = data.get('Poster', 'N/A')
                return {'title': title, 'release_date': release_date, 'overview': overview, 'poster_path': poster_path}
    return {'title': 'N/A', 'release_date': 'N/A', 'overview': 'N/A', 'poster_path': 'N/A'}

# Initialize the Flask server and Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Movie Recommendation System"

# Define the home page layout
home_layout = html.Div(
    className="page-content",
    children=[
        html.H1("IMDB Movies Recommendation System", style={'textAlign': 'center'}),
    ]
)

# Define the user recommendation page layout
user_layout = html.Div(
    className="page-content",
    children=[
        dcc.Dropdown(
            id='user-id-dropdown',
            options=[{'label': str(user_id), 'value': user_id} for user_id in user_movies['userId'].unique() ],
            placeholder='Select your user ID',
            className='dropdown',
        ),
        dcc.Input(id='num-movies-input', type='number', placeholder='Number of movies', className='input',style={'color': 'white'}),
        html.Button('Enter', id='user-id-button', className='btn'),
        html.Div(id='user-recommendations-output', style={'marginTop': '20px'}),
        html.Div(id='user-top-watched-output', style={'marginTop': '20px'}),
        dcc.Link(html.Button('Back', className='btn'), href='/', style={'marginTop': '20px'})
    ]
)

# Define the movie recommendation page layout
movie_layout = html.Div(
    className="page-content",
    children=[
        dcc.Dropdown(
            id='movie-dropdown',
            options=[{'label': title, 'value': title} for title in movies_df['title'].unique()],
            placeholder='Select a movie',
            className='dropdown',
        ),
        html.Button('Enter', id='movie-button', className='btn'),
        html.Div(id='movie-recommendations-output', style={'marginTop': '20px'}),
        dcc.Link(html.Button('Back', className='btn'), href='/', style={'marginTop': '20px'})
    ]
)

# Define the navbar layout
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dcc.Link('Home', href='/', className='nav-link')),
        dbc.NavItem(dcc.Link('User Recommendation', href='/user', className='nav-link')),
        dbc.NavItem(dcc.Link('Movie Recommendation', href='/movie', className='nav-link')),
    ],
    brand='IMDB Movies Recommendation System',
    color='#141414',
    dark=True,
    className='navbar'
)

# Define the app layout with navbar
app.layout = html.Div([
    navbar,
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# Define the page callback
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == '/user':
        return user_layout
    elif pathname == '/movie':
        return movie_layout
    else:
        return home_layout

# Define the user recommendations callback
@app.callback(
    [Output('user-recommendations-output', 'children'),
     Output('user-top-watched-output', 'children')],
    [Input('user-id-button', 'n_clicks')],
    [State('user-id-dropdown', 'value'), State('num-movies-input', 'value')]
)
def update_user_recommendations(n_clicks, user_id, num_movies):
    if n_clicks is None:
        return "", ""

    try:
        recommendations, top_watched = recommend_movie(int(user_id), int(num_movies))
        movie_details = [fetch_movie_details(movie_id) for movie_id in recommendations]
        cards = [
            dbc.Card(
                [
                    dbc.CardImg(src=details['poster_path'], top=True, className='card-img-top'),
                    dbc.CardBody(
                        [
                            html.H5(details['title'], className='card-title',style={'font-size': '17px', 'text-align': 'center'}),
                            html.P(f"Release Date: {details['release_date']}", className='card-text',style={'font-size': '12px'}),
                            html.P(details['overview'], className='card-text',style={'font-size': '10px'}),
                        ]
                    )
                ],
                className='card'
            )
            for details in movie_details
        ]

        top_watched_details = [fetch_movie_details(movie_id) for movie_id in top_watched]
        top_watched_cards = [
            dbc.Card(
                [
                    dbc.CardImg(src=details['poster_path'], top=True, className='card-img-top'),
                    dbc.CardBody(
                        [
                            html.H5(details['title'], className='card-title',style={'font-size': '17px', 'text-align': 'center'}),
                            html.P(f"Release Date: {details['release_date']}", className='card-text',style={'font-size': '12px'}),
                            html.P(details['overview'], className='card-text',style={'font-size': '10px',}),
                        ]
                    )
                ],
                className='card'
            )
            for details in top_watched_details
        ]

        recommendations_section = html.Div(
            [
                html.H3("Top Recommended Movies for the User"),
                dbc.Row([dbc.Col(card, width=2) for card in cards], className='movie-cards', style={'display': 'flex', 'flexWrap': 'nowrap', 'overflowX': 'auto'}),
            ]
        )

        top_watched_section = html.Div(
            [
                html.H3("User's Top 10 Watched Movies"),
                dbc.Row([dbc.Col(card, width=2) for card in top_watched_cards], className='movie-cards', style={'display': 'flex', 'flexWrap': 'nowrap', 'overflowX': 'auto'}),
            ]
        )

        return recommendations_section, top_watched_section

    except Exception as e:
        return f"Error: {str(e)}", ""

# Define the movie recommendations callback
@app.callback(
    Output('movie-recommendations-output', 'children'),
    [Input('movie-button', 'n_clicks')],
    [State('movie-dropdown', 'value')]
)
def update_movie_recommendations(n_clicks, movie_name):
    if n_clicks is None:
        return ""
    similar_movies = recommend_movies(movie_name)
    movie_details = [fetch_movie_details(links_df.loc[links_df['movieId'] == movie_id, 'imdbId'].values[0]) for movie_id in similar_movies]
    cards = [
        dbc.Card(
            [
                dbc.CardImg(src=details['poster_path'], top=True, className='card-img-top'),
                dbc.CardBody(
                    [
                        html.H5(details['title'], className='card-title',style={'font-size': '17px', 'text-align': 'center'}),
                        html.P(f"Release Date: {details['release_date']}", className='card-text',style={'font-size': '12px'}),
                        html.P(details['overview'], className='card-text',style={'font-size': '8px' }),
                    ]
                )
            ],
            className='card'
        )
        for details in movie_details
    ]
    return html.Div(
        dbc.Row([dbc.Col(card, width=2) for card in cards], className='movie-cards', style={'display': 'flex', 'flexWrap': 'nowrap', 'overflowX': 'auto'}),
    )

if __name__ == '__main__':
    app.run_server(debug=True)


