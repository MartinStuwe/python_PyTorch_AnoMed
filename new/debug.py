from datasets import load_dataset

# Load the dataset
data_dir = "./data"
celeba = load_dataset("tpremoli/CelebA-attrs", split="all", cache_dir=data_dir)

# Convert to a pandas DataFrame (if needed)
df = celeba.to_pandas()

# Extract attribute names (assuming the first column is 'image' and attributes start from the second column)
attribute_names = df.columns[1:-1]  # Exclude 'image' column and any non-attribute columns at the end

# Print attribute names with their corresponding indices
for idx, attr in enumerate(attribute_names):
    print(f"Index: {idx}, Attribute: {attr}")
