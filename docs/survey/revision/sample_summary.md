# Survey samples (built 2026-10-19)

## Bundestag chunks

| Party | n | EMI mean | EMI min | EMI max | words median |
| --- | --- | --- | --- | --- | --- |
| AFD | 80 | -0.13 | -3.74 | +2.62 | 65 |
| Alliance 90/The Greens | 55 | +0.02 | -2.65 | +3.25 | 68 |
| CDU/CSU | 20 | -0.10 | -2.72 | +1.60 | 66 |
| FDP | 50 | -0.00 | -2.70 | +3.59 | 68 |
| SPD | 20 | -0.02 | -3.06 | +2.72 | 68 |
| The Left Party | 75 | -0.02 | -3.13 | +2.52 | 67 |
| **All** | 300 | -0.04 | -3.74 | +3.59 | 67 |

## Tweets

| Party | n | EMI mean | EMI min | EMI max | words median |
| --- | --- | --- | --- | --- | --- |
| AFD | 50 | -0.11 | -1.85 | +1.78 | 30 |
| Alliance 90/The Greens | 50 | -0.02 | -1.30 | +2.03 | 30 |
| CDU/CSU | 50 | -0.03 | -1.93 | +2.80 | 29 |
| FDP | 50 | -0.07 | -1.78 | +2.09 | 31 |
| SPD | 50 | -0.07 | -1.76 | +1.66 | 26 |
| The Left Party | 50 | -0.10 | -1.78 | +1.17 | 29 |
| **All** | 300 | -0.07 | -1.93 | +2.80 | 30 |

## Build log

```
parliament: 59,017 speeches 2015-2025 in the six parties
parliament chunks from 24,000 speeches: 225,529 chunks, kept 145,058; dropped by reason {'ok': 145058, 'greeting': 22820, 'greeting_start': 22571, 'length': 18131, 'procedural': 9831, 'format': 4103, 'caps': 2064, 'longtoken': 425, 'fragment_start': 414, 'end': 91, 'start': 21}
parliament: 22,721 candidate chunks (one per speech) scored; chunk EMI vs speech EMI rho 0.64
parliament: 300 instances; per party x quintile:
q                        1   2   3   4   5
party                                     
AFD                     16  16  16  16  16
Alliance 90/The Greens  11  11  11  11  11
CDU/CSU                  4   4   4   4   4
FDP                     10  10  10  10  10
SPD                      4   4   4   4   4
The Left Party          15  15  15  15  15
parliament: wrote 300 instances to C:\Users\peers\working\phd\paper\nhb_emi\code\revision\survey\parliament\data_files (words median 67, min 46, max 89)
twitter: 1,390,846 original tweets 2015-2025 in the six parties (no replies, no retweets)
twitter pool 36,000 tweets: kept 5,287; dropped by reason {'short': 9891, 'mentions': 9014, 'end': 6758, 'ok': 5287, 'hashtags': 2230, 'fragment_or_headline': 1277, 'announcement': 678, 'language': 490, 'emoji': 218, 'retweet_or_reply': 118, 'caps': 21, 'format': 18}
twitter: 300 instances; per party x quintile:
q                        1   2   3   4   5
party                                     
AFD                     10  10  10  10  10
Alliance 90/The Greens  10  10  10  10  10
CDU/CSU                 10  10  10  10  10
FDP                     10  10  10  10  10
SPD                     10  10  10  10  10
The Left Party          10  10  10  10  10
twitter: wrote 300 instances to C:\Users\peers\working\phd\paper\nhb_emi\code\revision\survey\twitter\data_files (words median 30, min 15, max 47)
```

## Pilot (added 2026-10-19)

10 Bundestag chunks were moved from the parliament sample into `parliament_pilot/` (AfD 3, Left 2, Greens 2, FDP 1, CDU/CSU 1, SPD 1, spread over each party's EMI range; ids pilot_001, 019, 026, 082, 158, 187, 209, 231, 261, 291). The 10 pilot instances were merged back on 2026-10-20: the main parliament sample (local and deploy) again has all 300 instances; the pilot folder keeps its own copy of the 10 for the pilot run.
