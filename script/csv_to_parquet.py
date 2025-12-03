import pandas as pd
import os

def csv_to_parquet(input_csv_path: str, output_parquet_path: str) -> None:
	"""
	Convert a CSV file to Parquet format.

	Parameters:
	- input_csv_path: str : Path to the input CSV file.
	- output_parquet_path: str : Path to save the output Parquet file.
	"""
	# Read the CSV file into a DataFrame
	df = pd.read_csv(input_csv_path)

	# Save the DataFrame to Parquet format
	df.to_parquet(output_parquet_path, engine='pyarrow', index=False)

	csv_size = os.path.getsize(input_csv_path) / (1024 * 1024)  # Size in MB
	parquet_size = os.path.getsize(output_parquet_path) / (1024 * 1024)  # Size in MB

	print("Conversion complete!")
	print(f"CSV size: {csv_size:.2f} MB")
	print(f"Parquet size: {parquet_size:.2f} MB")

csv_path = ("./datasets/rees46_customer_model.csv")
parquet_path = ("./datasets/rees46_customer_model.parquet")
csv_to_parquet(csv_path, parquet_path)