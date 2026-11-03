# Third-party dictionaries used in the revision (discriminant validity, `code/05_validation/document/0519_discriminant_validity.py`)

| File | Content | Source and licence |
|---|---|---|
| `mMPD_de.yml` | German file of the multilingual Moral Political Dictionary: five moral foundations with virtue and vice poles, GENERAL, POSITIVE, NEGATIVE and MORALITY (union) | Simonsen, K. B. & Widmann, T. (2025). When do political parties moralize? A cross-national study of the use of moral language in political communication on immigration. *British Journal of Political Science*, 55. https://github.com/tweedmann/mMPD (MIT licence; cite the article) |
| `ed8.yml` (not included) | ed8 dictionary of eight discrete emotions in German political text (anger, fear, disgust, sadness, joy, enthusiasm, pride, hope) | Widmann, T. & Wich, M. (2023). Creating and comparing dictionary, word embedding, and transformer-based models to measure discrete emotions in German political text. *Political Analysis*, 31(4), 560–575. https://github.com/tweedmann/3x8emotions ("free to use for academic research"; cite the article). The script downloads `01_ed8/ed8.yml` from that repository into this folder on first use. |

Both dictionaries are lists of exact word forms (no wildcards); the scripts count hits per 100 lower-cased tokens.
