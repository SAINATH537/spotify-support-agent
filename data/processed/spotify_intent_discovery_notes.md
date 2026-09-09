# SpotifyCares Intent Discovery Notes

Dataset: spotify_customer_brand_pairs.csv
Customer→brand pairs: 43,092

## Candidate top-level taxonomy
1. Account / Login / Security
2. Premium / Subscription
3. Billing / Payment
4. Playback / Audio
5. App / Device / Technical
6. Playlist / Library
7. Content Availability
8. Ads
9. Feature / Product Question
10. Other / Ambiguous

## Important annotation rule
Assign the intent according to the customer's primary requested outcome, not every keyword present.

Examples:
- "I paid for Premium but my account still says Free" → Premium / Subscription
- "My card was charged twice" → Billing / Payment
- "Shuffle stops after one song" → Playback / Audio
- "The iPhone app crashes on launch" → App / Device / Technical
- "I can't log into my account" → Account / Login / Security
- "A song disappeared from Spotify" → Content Availability

## Caveat
The prevalence table is based on keyword matching only. Categories overlap heavily, so the counts are useful for discovery but must not be reported as labelled class frequencies.

## Next step
Create a stratified manual annotation set, refine boundaries between overlapping intents, then freeze the final taxonomy before training classifiers.
