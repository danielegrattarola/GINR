# Create data for first experiment
python data_generation/bunny.py
python data_generation/protein.py
python data_generation/us_election.py

# Create data for conditional INR experiments
python data_generation/bunny_time.py
python data_generation/bunny_time.py --full
python data_generation/protein_multiple.py

# Create data for weather experiments (this can take a couple of hours)
python data_generation/weather.py
python data_generation/weather_time.py
python data_generation/weather_time_sr.py