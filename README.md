# Fish-Identifier
This will be trained to identify fish of all species

## Data collection

Use the asynchronous scraper to download raw training images while respecting
provider rate limits and robots.txt rules:

```bash
python -m data_collection.scrape --species-list data/species.txt --provider duckduckgo
```

Images and metadata are stored under `data/raw/<provider>/<species>/` and audit
logs are written to `data/logs/`.
