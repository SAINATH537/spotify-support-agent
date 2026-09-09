import pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('data/processed/spotify_interactions.csv')
dupes = df['brand_response'].duplicated(keep=False).sum()
total = len(df)
print(f'Total interactions: {total}')
print(f'Duplicate brand_response rows: {dupes} ({dupes/total*100:.1f}%)')
print(f'Unique brand_responses: {df["brand_response"].nunique()}')
print()

top_repeated = df['brand_response'].value_counts().head(5)
print('Most repeated brand responses:')
for resp, cnt in top_repeated.items():
    print(f'  [{cnt}x] {resp[:90]}')
print()

golden = pd.read_csv('data/golden/spotify_golden_set_annotation_template_250.csv')
print('Golden set columns:', golden.columns.tolist())
print('Golden set shape:', golden.shape)
golden_ids = set(golden['tweet_id'].astype(str))
indexed_ids = set(df['customer_tweet_id'].astype(str))
overlap = golden_ids & indexed_ids
print(f'Golden tweet IDs: {len(golden_ids)}')
print(f'Indexed tweet IDs: {len(indexed_ids)}')
print(f'Overlap (MUST be 0): {len(overlap)}')
if overlap:
    print('  WARNING - OVERLAP IDs:', list(overlap)[:5])
else:
    print('  OK: No overlap between golden set and indexed interactions.')
