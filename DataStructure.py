"""
Movie and Actor Recommendation System with Graph-Based Analysis

This script construct graphs among movies and actors. 
And it leverages graph-based analysis for tasks such as recommending similar movies, 
identifying influential actors, and visualizing actor connections.

Dependencies:
- pandas
- numpy
- networkx
- matplotlib
- textwrap

Classes:
    1. MovieData:
        Handles the loading, cleaning, and aggregation of movie and actor datasets. 
        Includes functionality for normalizing popularity scores.

    2. GraphBuilder:
        Builds actor and movie graphs based on co-starring relationships and genre similarity.
        Supports finding influential actors and shortest paths between actors.

    3. Recommender:
        Provides recommendation functionalities, such as similar movies and actor-based searches.

Functions:
    - visualize_direct_links: Visualizes the top N direct connections for a node in a graph.

"""

import matplotlib
matplotlib.use('Agg')  # Set the backend
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import textwrap

class MovieData:
    """
    Handles loading, cleaning, and preprocessing of movie and actor datasets.

    Attributes:
        file_path (str): Path to the CSV file containing movie and actor data.
        df (pd.DataFrame): Raw loaded dataset.
        movies_df (pd.DataFrame): Aggregated movie-level dataset.
        actors_df (pd.DataFrame): Actor-level dataset.
        actor_id_to_name (dict): Mapping of actor IDs to their names.
        min_popularity (float): Minimum popularity value in the dataset.
        max_popularity (float): Maximum popularity value in the dataset.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.movies_df = None
        self.actors_df = None
        self.actor_id_to_name = {}
        self.min_popularity = None
        self.max_popularity = None

    def load_data(self):
        """
        Loads data from the specified CSV file.

        Returns:
            MovieData: The current instance of MovieData (for method chaining).
        """
        self.df = pd.read_csv(self.file_path)
        return self

    def clean_data(self):
        self.df['homepage'] = self.df['homepage'].fillna('')
        self.df = self.df.dropna(subset=['title', 'actor_name'])
        return self

    def aggregate_data(self):
        """
        Aggregates the dataset into movie-level and actor-level datasets.

        - Groups movie data to create a detailed movie-level dataset.
        - Creates an actor-level dataset with unique actors.
        - Normalizes popularity values for movies.

        Returns:
            MovieData: The current instance of MovieData (for method chaining).
        """
        self.movies_df = (
            self.df.groupby('movie_id')
            .agg({
                'title': 'first',
                'genres': 'first',
                'overview': 'first',
                'vote_average': 'first',
                'vote_count': 'first',
                'release_date': 'first',
                'homepage': 'first',
                'movie_popularity': 'first'
            })
            .reset_index()
        )

        actors_per_movie = (
            self.df.groupby('movie_id')['actor_id']
            .apply(list)
            .reset_index(name='actor_ids')
        )
        self.movies_df = self.movies_df.merge(actors_per_movie, on='movie_id')
        self.actors_df = self.df[['actor_id', 'actor_name']].drop_duplicates()
        self.actor_id_to_name = dict(zip(self.actors_df['actor_id'], self.actors_df['actor_name']))
        self.movies_df['year'] = pd.to_datetime(self.movies_df['release_date']).dt.year
        self.min_popularity = self.movies_df['movie_popularity'].min()
        self.max_popularity = self.movies_df['movie_popularity'].max()
        return self

    def normalize_popularity(self, p):
        """
        Normalizes a popularity value to the range [0, 1].

        Args:
            p (float): The popularity value to normalize.

        Returns:
            float: The normalized popularity value.
        """
        if self.min_popularity == self.max_popularity:
            return 0.5
        return (p - self.min_popularity) / (self.max_popularity - self.min_popularity)

class GraphBuilder:
    """
    Builds and manages actor and movie graphs based on relationships and attributes.

    Attributes:
        movies_df (pd.DataFrame): Aggregated movie-level dataset.
        actors_df (pd.DataFrame): Actor-level dataset.
        actor_id_to_name (dict): Mapping of actor IDs to names.
        data (MovieData): Reference to a MovieData instance for normalization.
        actor_graph (nx.Graph): Graph representing actor relationships.
        movie_graph (nx.Graph): Graph representing movie relationships.
    """
    def __init__(self, movies_df, actors_df, actor_id_to_name, data):
        self.movies_df = movies_df
        self.actors_df = actors_df
        self.actor_id_to_name = actor_id_to_name
        self.data = data  # Pass MovieData instance to access normalization
        self.actor_graph = nx.Graph()
        self.movie_graph = nx.Graph()

    def build_actor_graph(self):
        """
        Constructs a weighted graph of actor relationships.

        - Nodes represent actors.
        - Edges represent co-starring relationships, weighted by movie popularity.

        Returns:
            nx.Graph: The constructed actor graph.
        """

        # Add actor nodes
        for idx, row in self.actors_df.iterrows():
            self.actor_graph.add_node(row['actor_id'], name=row['actor_name'])

        # Add edges between actors who co-starred in the same movie
        # Weight now depends on normalized popularity of the movie
        for idx, row in self.movies_df.iterrows():
            actor_list = row['actor_ids']
            movie_pop = row['movie_popularity']
            norm_pop = self.data.normalize_popularity(movie_pop)
            increment = 1 + norm_pop  # Edge weight increment
            movie_title = row['title']  # Get the current movie's title

            for i in range(len(actor_list)):
                for j in range(i+1, len(actor_list)):
                    a1, a2 = actor_list[i], actor_list[j]

                    if self.actor_graph.has_edge(a1, a2):
                        # Update existing edge
                        self.actor_graph[a1][a2]['weight'] += increment
                        self.actor_graph[a1][a2]['movies'].add(movie_title)
                    else:
                        # Create a new edge with weight and a set of movies
                        self.actor_graph.add_edge(a1, a2, weight=increment, movies={movie_title})

        return self.actor_graph


    def get_most_influential_actor(self, start_year=None, end_year=None):
        """
        Identifies the most influential actor within a specified year range using PageRank.

        Args:
            start_year (int, optional): The start year for filtering movies.
            end_year (int, optional): The end year for filtering movies.

        Returns:
            tuple: (actor_id, actor_name, PageRank score) of the most influential actor.
        """
        if start_year is not None and end_year is not None:
            yearly_movies = self.movies_df[
                (self.movies_df['year'] >= start_year) & (self.movies_df['year'] <= end_year)
            ]
        else:
            yearly_movies = self.movies_df

        yearly_graph = nx.Graph()

        actor_ids_for_range = set(aid for actor_ids in yearly_movies['actor_ids'] for aid in actor_ids)
        for actor_id in actor_ids_for_range:
            a_name = self.actor_id_to_name.get(actor_id, "Unknown")
            yearly_graph.add_node(actor_id, name=a_name)

        for idx, row in yearly_movies.iterrows():
            actor_list = row['actor_ids']
            # Reuse same weighting scheme as in the actor graph
            movie_pop = row['movie_popularity']
            norm_pop = self.data.normalize_popularity(movie_pop)
            increment = 1 + norm_pop
            for i in range(len(actor_list)):
                for j in range(i+1, len(actor_list)):
                    a1, a2 = actor_list[i], actor_list[j]
                    if yearly_graph.has_edge(a1, a2):
                        yearly_graph[a1][a2]['weight'] += increment
                    else:
                        yearly_graph.add_edge(a1, a2, weight=increment)

        if len(yearly_graph.nodes) == 0:
            print(f"No data available for years {start_year}-{end_year}.")
            return None, None, None

        pr = nx.pagerank(yearly_graph, weight='weight')
        most_influential = max(pr, key=pr.get)
        return most_influential, self.actor_id_to_name[most_influential], pr[most_influential]

    def find_shortest_actor_connection(self, actor_name_1, actor_name_2):
        """
        Finds the shortest connection between two actors in the actor graph.

        Args:
            actor_name_1 (str): Name of the first actor.
            actor_name_2 (str): Name of the second actor.

        Returns:
            tuple: (list of actor names in the shortest path, explanation text).
        """
        # Convert actor names to lowercase
        actor_name_1 = actor_name_1.strip().lower()
        actor_name_2 = actor_name_2.strip().lower()

        # Look up actor IDs
        actor_row_1 = self.actors_df[self.actors_df['actor_name'].str.lower() == actor_name_1]
        actor_row_2 = self.actors_df[self.actors_df['actor_name'].str.lower() == actor_name_2]

        if actor_row_1.empty or actor_row_2.empty:
            return None, "One or both actors not found in the dataset."

        actor_id_1 = actor_row_1['actor_id'].iloc[0]
        actor_id_2 = actor_row_2['actor_id'].iloc[0]

        # Check if actor IDs exist in the actor graph
        if actor_id_1 not in self.actor_graph or actor_id_2 not in self.actor_graph:
            return None, "One or both actors not found in the actor graph."

        try:
            # Compute the shortest path
            shortest_path_ids = nx.shortest_path(self.actor_graph, source=actor_id_1, target=actor_id_2)
        except nx.NetworkXNoPath:
            return None, "No connection between the specified actors."

        shortest_path_names = [self.actor_id_to_name[aid] for aid in shortest_path_ids]

        # Construct explanation text for each level
        explanation_lines = []
        for i in range(len(shortest_path_ids) - 1):
            a1 = shortest_path_ids[i]
            a2 = shortest_path_ids[i + 1]
            edge_data = self.actor_graph.get_edge_data(a1, a2)
            movies = edge_data.get('movies', [])
            a1_name = f'{self.actor_id_to_name[a1]}'
            a2_name = f'{self.actor_id_to_name[a2]}'

            if movies:
                movie_list = " / ".join([f"{movie}" for movie in movies])
                explanation_lines.append(f"{a1_name} and {a2_name} both act in movies: {movie_list}")
            else:
                explanation_lines.append(f"{a1_name} and {a2_name} connection found (no movie list available)")

        explanation_text = "<br>".join(explanation_lines)
        return shortest_path_names, explanation_text



    def build_movie_graph(self, alpha=1.0, beta=0.5, gamma=1.0, delta=0.5):
        """
        Constructs a weighted graph of movie relationships.

        - Nodes represent movies.
        - Edges represent similarity based on shared actors and genres, adjusted by popularity.

        Args:
            alpha (float): Weight for shared actors.
            beta (float): Weight for shared genres.
            gamma (float): Scaling factor for popularity contributions.
            delta (float): Adjustment factor for the popularity boost.

        Returns:
            nx.Graph: The constructed movie graph.
        """

        for idx, row in self.movies_df.iterrows():
            self.movie_graph.add_node(row['movie_id'],
                                      title=row['title'],
                                      genres=row['genres'],
                                      actors=row['actor_ids'],
                                      popularity=row['movie_popularity'])

        movie_id_to_actors = dict(zip(self.movies_df['movie_id'], self.movies_df['actor_ids']))
        movie_id_to_genres = dict(zip(self.movies_df['movie_id'], self.movies_df['genres']))
        movie_id_to_popularity = dict(zip(self.movies_df['movie_id'], self.movies_df['movie_popularity']))

        # For genre sets
        movie_id_to_genreset = {mid: set(g.lower().split('|')) for mid, g in movie_id_to_genres.items() if isinstance(g, str)}

        movie_ids = self.movies_df['movie_id'].tolist()

        for i in range(len(movie_ids)):
            for j in range(i+1, len(movie_ids)):
                m1 = movie_ids[i]
                m2 = movie_ids[j]
                actors1 = set(movie_id_to_actors[m1])
                actors2 = set(movie_id_to_actors[m2])
                genres1 = movie_id_to_genreset.get(m1, set())
                genres2 = movie_id_to_genreset.get(m2, set())

                shared_actors = len(actors1.intersection(actors2))
                shared_genres = len(genres1.intersection(genres2))

                if shared_actors > 0 or shared_genres > 0:
                    base_sim = alpha * shared_actors + beta * shared_genres

                    # Incorporate popularity if both movies are popular
                    p1 = self.data.normalize_popularity(movie_id_to_popularity[m1])
                    p2 = self.data.normalize_popularity(movie_id_to_popularity[m2])

                    # If both have high popularity, give a boost
                    # Example: similarity *= (1 + delta * sqrt(p1 * p2))
                    # gamma may be used to scale the entire popularity contribution
                    pop_boost = (1 + delta * np.sqrt(p1 * p2))  # If one or both are not popular, p1 or p2 is small, so the boost is small.

                    sim_score = base_sim * pop_boost * gamma
                    self.movie_graph.add_edge(m1, m2, weight=sim_score)

        return self.movie_graph

class Recommender:
    """
    Provides recommendation functionalities based on actor and movie graphs.

    Attributes:
        movies_df (pd.DataFrame): Aggregated movie-level dataset.
        actors_df (pd.DataFrame): Actor-level dataset.
        actor_id_to_name (dict): Mapping of actor IDs to names.
        actor_graph (nx.Graph): Graph of actor relationships.
        movie_graph (nx.Graph): Graph of movie relationships.
        title_to_id (dict): Mapping of movie titles to their IDs.
    """

    def __init__(self, movies_df, actors_df, actor_id_to_name, actor_graph, movie_graph):
        self.movies_df = movies_df
        self.actors_df = actors_df
        self.actor_id_to_name = actor_id_to_name
        self.actor_graph = actor_graph
        self.movie_graph = movie_graph

        self.title_to_id = dict(zip(self.movies_df['title'].str.lower(), self.movies_df['movie_id']))

    def search_movies(movies_df, actors_df, actor_id_to_name, factor, query, sort_by, ascending):
        """
        Searches for movies based on a factor (title, actor, or genre).

        Args:
            factor (str): The factor to search by ('title', 'actor', 'genre').
            query (str): The search query.
            sort_by (str): The sorting criterion.
            ascending (bool): Whether to sort in ascending order.

        Returns:
            pd.DataFrame: A DataFrame of search results.
        """
        # Normalize the query string
        query = query.lower().strip()
        
        # Define the mapping for sort columns
        sort_columns = {
            "release time": "release_date",
            "average vote rate": "vote_average",
            "vote count": "vote_count",
            "popularity": "movie_popularity"
        }

        # Validate input parameters
        if factor not in ["title", "actor", "genre"]:
            raise ValueError("Invalid search factor. Choose 'title', 'actor', or 'genre'.")
        if sort_by not in sort_columns:
            raise ValueError("Invalid sort option. Choose a valid sorting criterion.")
        if not query:
            raise ValueError("Query cannot be empty.")

        # Initialize an empty DataFrame for results
        result_df = pd.DataFrame()

        if factor == "title":
            # Search titles with partial match, handle NaN values
            result_df = movies_df[movies_df['title']
                                .fillna('')
                                .str.lower()
                                .str.contains(query, na=False)]
        elif factor == "actor":
            # Search actors with partial match, handle NaN values
            actor_row = actors_df[actors_df['actor_name']
                                .fillna('')
                                .str.lower()
                                .str.contains(query, na=False)]
            if not actor_row.empty:
                # Assuming you want to include all actors that match the query
                # If only the first match is desired, keep iloc[0]
                actor_ids = actor_row['actor_id'].tolist()
                # Filter movies where any of the actor_ids are in the movie's actor_ids list
                result_df = movies_df[movies_df['actor_ids']
                                    .apply(lambda x: any(actor_id in x for actor_id in actor_ids) if isinstance(x, list) else False)]
        elif factor == "genre":
            # Search genres with partial match, handle NaN values
            result_df = movies_df[movies_df['genres']
                                .fillna('')
                                .str.lower()
                                .str.contains(query, na=False)]
        
        # If no results found, return the empty DataFrame
        if result_df.empty:
            return result_df

        # Determine the column to sort by
        sort_column = sort_columns[sort_by]

        # Verify that the sort column exists in the DataFrame
        if sort_column not in result_df.columns:
            raise ValueError(f"Sort column '{sort_column}' does not exist in the DataFrame.")

        # Sort the results
        return result_df.sort_values(by=sort_column, ascending=ascending)



    def recommend_similar_movies(self, movie_title, top_n=10):
        """
        Recommends movies similar to a specified movie.

        Args:
            movie_title (str): Title of the movie to find recommendations for.
            top_n (int): Number of recommendations to return.

        Returns:
            list: Recommended movies and their similarity scores.
        """
        movie_title_lower = movie_title.lower()
        if movie_title_lower not in self.title_to_id:
            return []
        mid = self.title_to_id[movie_title_lower]
        if mid not in self.movie_graph:
            return []

        neighbors = self.movie_graph[mid]
        scored_neighbors = [(nmid, data['weight']) for nmid, data in neighbors.items()]
        scored_neighbors.sort(key=lambda x: x[1], reverse=True)

        recommended = []
        for nmid, score in scored_neighbors[:top_n]:
            recommended.append((self.movie_graph.nodes[nmid]['title'], score))
        return recommended


def generate_movie_summary(client, movie_name, genres, overview, vote_average):
    """
    Generates a concise summary for a movie using OpenAI's API.

    Args:
        client (OpenAI): OpenAI client instance.
        movie_name (str): Name of the movie.
        genres (str): Genres of the movie.
        overview (str): Overview of the movie.
        vote_average (float): Average rating of the movie.

    Returns:
        str: The generated movie summary.
    """
    prompt = (
        f"Write a concise summary about the movie '{movie_name}' in less than 100 words. "
        f"Genres: {genres}. It has an average rating of {vote_average}. "
        f"Here is the overview: {overview}"
        f"Keep the output format relatively fixed. Talk about the genres, vote_average and then a concise description of overview"
    )
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant specializing in movies."},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content

def visualize_direct_links(
    graph,
    main_node_id,
    top_n=10,
    filename="static/connections.png",
    get_label_func=None,
    node_color="darkblue",
    edge_color="white",
    background_color="black",
    main_node_size=1200,
    base_size=300,
    max_size=1000,
    wrap_width=20  # Add a parameter for wrapping width
):
    """
    Visualizes the top N direct connections of a given node in a graph.

    Args:
        graph (nx.Graph): The graph to visualize.
        main_node_id: ID of the main node.
        top_n (int): Number of top connections to display.
        filename (str): Path to save the visualization image.
        get_label_func (callable): Function to generate labels for nodes.
        node_color (str): Color of the nodes.
        edge_color (str): Color of the edges.
        background_color (str): Background color of the plot.
        main_node_size (int): Size of the main node.
        base_size (int): Base size for neighbor nodes.
        max_size (int): Maximum size for neighbor nodes.
        wrap_width (int): Width for wrapping labels.

    Returns:
        bool: True if visualization is successful, False otherwise.
    """

    # Ensure top_n is an integer
    try:
        top_n = int(top_n)
    except (ValueError, TypeError):
        top_n = 10  # Default to 10 if invalid or not provided

    if main_node_id not in graph:
        print(f"Main node ID {main_node_id} not found in the graph.")
        return False

    # Extract neighbors and weights
    connections = [(neighbor, data['weight']) for neighbor, data in graph[main_node_id].items()]
    if not connections:
        print(f"No connections found for main node ID {main_node_id}.")
        return False

    # Take top N connections
    top_connections = sorted(connections, key=lambda x: x[1], reverse=True)[:top_n]

    # Build subgraph
    subgraph = nx.Graph()
    subgraph.add_node(main_node_id)
    max_weight = top_connections[0][1] if top_connections else 1.0

    for neighbor, weight in top_connections:
        subgraph.add_node(neighbor)
        subgraph.add_edge(main_node_id, neighbor, weight=weight)

    pos = nx.spring_layout(subgraph, seed=42, k=0.5)

    # Ensure a label function is provided, fallback to default
    if get_label_func is None:
        get_label_func = lambda n: graph.nodes[n].get('name', str(n))  # Use 'name' from the graph node

    def wrap_label(text, width):
        # Wrap text into multiple lines if it exceeds the given width
        return "\n".join(textwrap.wrap(text, width=width))

    # Wrap the labels
    labels = {n: wrap_label(get_label_func(n), wrap_width) for n in subgraph.nodes()}

    # Compute node sizes
    node_sizes = []
    for node in subgraph.nodes():
        if node == main_node_id:
            node_sizes.append(main_node_size)
        else:
            w = subgraph[main_node_id][node]['weight']
            scaled_size = base_size + (w / max_weight) * (max_size - base_size)
            node_sizes.append(scaled_size)

    plt.figure(figsize=(12, 10), facecolor=background_color)
    nx.draw(
        subgraph,
        pos,
        with_labels=True,
        labels=labels,
        node_color=node_color,
        edge_color=edge_color,
        font_size=12,
        font_color="white",
        font_weight="bold",  # Set font to bold
        node_size=node_sizes,
        width=2.0
    )

    edge_labels = nx.get_edge_attributes(subgraph, "weight")
    formatted_edge_labels = {k: f"{v:.2f}" for k, v in edge_labels.items()}
    nx.draw_networkx_edge_labels(
        subgraph,
        pos,
        edge_labels=formatted_edge_labels,
        font_size=10,
        label_pos=0.5,
        font_color="#f1f4db",
        rotate=False,
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="none", facecolor="#375762")  # Custom text box
    )

    main_node_label = labels[main_node_id]
    plt.title(f"Top {top_n} Direct Connections for Node: {main_node_label}", fontsize=20, color="pink")
    plt.axis("off")
    plt.savefig(filename, bbox_inches="tight", pad_inches=0.5, facecolor=background_color)
    plt.close()
    return True
