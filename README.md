# FuzzyTrees
FuzzyTrees is a machine-learning application built in Python and Hex that combines fuzzy-matching techniques with a random forest classifier to do approximate matches on records from separate tables/data sources. In the original use case, this was to find duplicate vet practice records that were flowing in through DBT from vso and vss data sources all the way into our integrated online practice table in our core DBT layer. Duplicate records within our infrastructure can compromise the integrity of downstream reporting and analytics, this piece of work represents an automation tool for identifying duplicates, as well as performing a general fuzzy-matching functionality.

![image](https://github.com/user-attachments/assets/14acbb26-73f4-4ffd-b4ec-5240f2a0896e)

For all pairs of records in two tables, FuzzyTrees computes similarity metrics on shared fields (name, city, address, zip) using the <a href='https://en.wikipedia.org/wiki/Levenshtein_distance'>Levenshtein distance</a> and <a href='https://en.wikipedia.org/wiki/Jaro%E2%80%93Winkler_distance'>Jaro-Winkler distance</a>. These similarity metrics are 
used for feature engineering for the random forest classifier algorithm. Scanning over duplicate VSS practices, FuzzyTrees finds the best approximate match in the table of VSO practices--providing a base mapping for systematic deduplication. 

![image](https://github.com/user-attachments/assets/7b93cc79-6190-4a97-b09e-bb2d3d710b80)

![image](https://github.com/user-attachments/assets/877b911d-ca0c-4947-baf2-a6b2237b1825)
