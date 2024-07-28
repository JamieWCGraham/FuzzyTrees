# FuzzyTrees
FuzzyTrees is a machine-learning application built in Python and Hex that combines fuzzy-matching techniques with a random forest classifier to do approximate matches on records from separate tables/data sources. In the original use case, this was to find duplicate vet practice records that were flowing in through DBT from vso and vss data sources all the way into our integrated online practice table in our core DBT layer. Duplicate records within our infrastructure can compromise the integrity of downstream reporting and analytics, this piece of work represents an automation tool for identifying duplicates, as well as performing a general fuzzy-matching functionality.

![image](https://github.com/user-attachments/assets/14acbb26-73f4-4ffd-b4ec-5240f2a0896e)

For all pairs of records in two tables, FuzzyTrees computes similarity metrics on shared fields (name, city, address, zip) using the <a href='https://en.wikipedia.org/wiki/Levenshtein_distance'>Levenshtein distance</a> and <a href='https://en.wikipedia.org/wiki/Jaro%E2%80%93Winkler_distance'>Jaro-Winkler distance</a>. These similarity metrics are 
used for feature engineering for the random forest classifier algorithm. Scanning over duplicate VSS practices, FuzzyTrees finds the best approximate match in the table of VSO practices--providing a base mapping for systematic deduplication. 

## Example


![image](https://github.com/user-attachments/assets/f7922de9-630d-48f3-8009-d670eeb33094)


Here is an example, the first row here is a matched pair of VSO and VSS practices. On the right of the table, on van view engineered similarity features, taking the Levenshtein and Jaro-Winkler-similarity measures (known as distances) with the name, city, address, and zip fields from the VSO and VSS practices, respectively. We take these 'fuzzy' similarity measures because we can't do an exact matching process for these records since the fields aren't exactly identical (one VSS practice may spell their address with "St.", the VSO practice spelling that same duplicated practice with "Street" instead). The specific distances have slightly different use cases--Jaro-Winkler is best for name matching, and texts with similar prefixes. Levenshtein is good for short texts, and for spelling corrections/typos.

These features tell us how similar the given fields are across the VSO and VSS data soures. We can observe that across the board for the first row, the name, city, address, and zipcodes between VSO and VSS data sources match perfectly, and so they obtain distance scores of 1. On the other hand, with the second row, the VSO and VSS practices do not match, and the similarity scores are much less than 1 across the board.

We cast the practice matching task as a binary classification problem, the algorithm should output 1 if the practices match, and 0 otherwise. The random forest classifier is a state of the art ML algorithm that averages a bunch of decision trees generated on subsamples of the data to determine this binary classification. In this sense, it determines how important each of these features are for the classification of a 'match'. For instance, it's okay if the addresses have slight differences (St. vs Street, etc.), but the zipcode should be a perfect match.

## Methodology and Evaluation

A base dataset of 128K pairs of practices was used, 4.4% of which represent true fuzzy matches. A 60/20/20 train/validation/test split was employed. The testing and validation sets consists of ~24K pairs of practices and ~1K non-matching pairs of practices. Validation set serves as a basis to understand the baseline generalizability of the model past the training data, and for tuning hyperparameters to improve performance--before testing predictions on the test set. Hyperparameter of n_estimators was scanned in a grid search and found to show no performance difference past n_estimators > 20. The test set is used to evaluate the model's performance after it has been trained and validated. The test set provides an unbiased evaluation of the final model's performance. It serves to simulate how the model would perform on unseen data in a real-world scenario. FuzzyTrees works extremely well on it's designed use-case for identifying practice duplicates with the currently prescribed features across data-sources, next steps are to extend the model for generalized fuzzy-matching across the organization and evaluate performance metrics in generalized scenarios, improve UI/UX features of the tool, as well as move towards productionizing the model in Iguazio. 


![image](https://github.com/user-attachments/assets/7b93cc79-6190-4a97-b09e-bb2d3d710b80)

![image](https://github.com/user-attachments/assets/877b911d-ca0c-4947-baf2-a6b2237b1825)

## UI/UX in Hex

![image](https://github.com/user-attachments/assets/245a790a-2656-4f79-8c40-c5736d4e00a3)

