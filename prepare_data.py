import os
import pandas as pd

print("Loading consumption and weather data...")
consumption_df = pd.read_csv("Data/consumption.csv", nrows=2000000) 
weather_df = pd.read_csv("Data/weather.csv")

print(f"Loaded consumption rows: {len(consumption_df)}")
print(f"Loaded weather rows: {len(weather_df)}")

# Clean and extract site_id
consumption_df['site_id'] = consumption_df['source_building_id'].astype(str).apply(lambda x: x.split('_')[0].strip())
weather_df['site_id'] = weather_df['site_id'].astype(str).str.strip()

# Clean timestamp formatting
consumption_df['timestamp_local'] = consumption_df['timestamp_local'].astype(str).str.strip()
weather_df['timestamp_local'] = weather_df['timestamp_local'].astype(str).str.strip()

print("Merging datasets on timestamp_local and site_id...")
merged_df = pd.merge(
    consumption_df, 
    weather_df, 
    on=["timestamp_local", "site_id"], 
    how="inner"
)

print(f"Rows after merge: {len(merged_df)}")

# Target only critical columns for dropping NaNs (instead of wiping out rows with minor missing optional weather metrics)
critical_columns = ['energy_kwh', 'airTemperature', 'timestamp_local', 'site_id']
merged_df = merged_df.dropna(subset=critical_columns)

# Optional: if you want to sample down slightly if it gets too large, or let it hit our ~80MB sweet spot:
# If 1.7M rows creates a file that is too large (e.g., > 90MB), we can sample it:
if merged_df.memory_usage(deep=True).sum() / (1024 * 1024) > 90:
    merged_df = merged_df.sample(n=800000, random_state=42)

print(f"Rows after targeted cleaning: {len(merged_df)}")

# Force cap the final rows to ensure the CSV file stays strictly under 50 MB
MAX_TARGET_ROWS = 400000
if len(merged_df) > MAX_TARGET_ROWS:
    print(f"📉 Subsampling from {len(merged_df)} to {MAX_TARGET_ROWS} rows to ensure file size stays under 50 MB...")
    merged_df = merged_df.sample(n=MAX_TARGET_ROWS, random_state=42)
    
# Save file
os.makedirs("Data", exist_ok=True)
output_path = "Data/energy_training_data.csv"
merged_df.to_csv(output_path, index=False)

file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
print(f"Unified training dataset created successfully! Size: {file_size_mb:.2f} MB")