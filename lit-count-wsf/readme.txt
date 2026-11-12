Fig S.5 literature count.

wsf_pubs_counts_by_year.csv: number of records per publication year returned by
a Web of Science Core Collection topic search (timespan 1637-2025; 91 records,
1944-2025). Only per-year counts are included; the exported bibliographic records
are not redistributed. wsf_pubs_fig.py plots 1950-2025 with a linear trend.

Query:
TS = (
("water supply forecast*" OR "seasonal streamflow forecast*" OR "snowmelt runoff forecast*" OR "water supply outlook" OR "water supply forecasts" OR "water supply forecasting")
AND
("Colorado" OR "Columbia River" OR "Snake River" OR "Pacific Northwest" OR "California" OR "Rio Grande" OR "Great Basin" OR "western United States" OR "western US")
) AND Preprint Citation Index (Exclude - Database) AND Research Commons (Exclude - Database)
Timespan: 1637-2025.
