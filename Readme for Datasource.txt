### Data Sources Overview

## Origin:

1.API Token: 
"eyJhbGciOiJIUzI1NiJ9eyJhdWQiOiI1Zjc2NjgzYzZiMzQ2MThlZTVjZTY4ZTU3M2ZlYzc0NyIsIm5iZiI6MTczMjQ4ODAyOS43MjcsInN1YiI6IjY3NDNhYjVkNjg4MjMwMDRjYTljYjJjMiIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.KmOn1hc0C0Q0VGZ8HxPraIeu9f9GvYUAK-MbiekEFK0"
2.Data Source: 
	1)TMDB API (The Movie Database), TMDB API Documentation: https://developer.themoviedb.org/docs
	2)Endpoints Used:
            Movie Details: https://api.themoviedb.org/3/movie/{movie_id}
            Movie Credits: https://api.themoviedb.org/3/movie/{movie_id}/credits

## Formats:
    Input Data Format: 
        JSON
    Output File Formats:
        CSV: movies_actors.csv
## Data Access and Caching:

	1.Access Method: The TMDB API was accessed using HTTP requests through the requests Python library.
	Rate Limiting: A delay of 0.25 seconds was added between API calls to respect TMDB's rate limits.

	2.To save time for launching project. I have saved data retrieved from TMDB API into movies_actors.csv file.
	This file contains most of popular movies (first 50 pages) released from 2015 to 2024 on TMDB database. If you want to retrieve even more movie data, which may 	spend long time, from TMDB API, you can modify release_year parameters in TMDB_Data_Retrieve.py. and run it. Then the exist movies_actors.csv file 
	will be replaced.

## Data Summary:
    1. Data Collected:
        1) popular_movies.json - Movies Data (from /movie/{movie_id}):

        Number of Variables: 8
        id (Movie ID)
        title (Movie Title)
        genres (Pipe-separated genres)
        homepage (Movie website)
        overview (Description of the movie)
        release_date (Movie release date)
        vote_average (Average rating)
        popularity (TMDB popularity score)

        2) movie_data.json - Combined Movies & Actors Data :

        Number of Variables: 13

        movie_id
        title
        genres
        homepage
        overview
        release_date
        vote_average
        vote_count
        movie_popularity
        actor_id
        actor_name
        actor_gender
        actor_popularity


        3) movies_actors.csv (Output Combined Data):

        Total Variables: 13

        movie_id
        title
        genres
        homepage
        overview
        release_date
        vote_average
        vote_count
        movie_popularity
        actor_id
        actor_name
        actor_gender
        actor_popularity
        

Notes:
1. Each row corresponds to an actor appearing in a movie. A movie with multiple actors will have multiple rows.
2. Popular movies were prioritized and retrieved using the popularity.desc sorting parameter, so the data covers most popular movies released between January 1, 2015, and December 31, 2024.
4. Only actors with known_for_department == "Acting" were included in the final output.