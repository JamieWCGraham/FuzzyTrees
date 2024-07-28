# FuzzyTrees
FuzzyTrees is a machine-learning application built in Python and Hex that combines fuzzy-matching techniques with a random forest classifier to do approximate matches on records from separate tables/data sources. In the original use case, this was to find duplicate vet practice records that were flowing in through DBT from vso and vss data sources all the way into our integrated online practice table in our core DBT layer. Duplicate records within our infrastructure can compromise the integrity of downstream reporting and analytics, this piece of work represents an automation tool for identifying duplicates, as well as performing a general fuzzy-matching functionality.

![image](https://github.com/user-attachments/assets/14acbb26-73f4-4ffd-b4ec-5240f2a0896e)

For all pairs of records in two tables, FuzzyTrees computes similarity metrics on shared fields (name, city, address, zip) using the <a href='https://en.wikipedia.org/wiki/Levenshtein_distance'>Levenshtein distance</a> and <a href='https://en.wikipedia.org/wiki/Jaro%E2%80%93Winkler_distance'>Jaro-Winkler distance</a>. These similarity metrics are 
used for feature engineering for the random forest classifier algorithm. Scanning over duplicate VSS practices, FuzzyTrees finds the best approximate match in the table of VSO practices--providing a base mapping for systematic deduplication. 

A base dataset of 128K pairs of practices was used, 4.4% of which represent true fuzzy matches. A 60/20/20 train/validation/test split was employed. The testing and validation sets consists of ~24K pairs of practices and ~1K non-matching pairs of practices. Validation set serves as a basis to understand the baseline generalizability of the model past the training data, and for tuning hyperparameters to improve performance--before testing predictions on the actual test set. Hyperparameter of n_estimators was scanned in a grid search and found to show no performance difference past n_estimators > 20. The test set is used to evaluate the model's performance after it has been trained and validated. The key characteristic of the test set is that it is only used once, at the very end of the model development process, to provide an unbiased evaluation of the final model's performance. It simulates how the model would perform on unseen data in a real-world scenario. FuzzyTrees works very well on it's designed use-case, next steps are to extend the model for generalized fuzzy-matching across the organization, and improve UI/UX features of the tool, as well as move towards productionizing the model in Iguazio. 


![image](https://github.com/user-attachments/assets/7b93cc79-6190-4a97-b09e-bb2d3d710b80)

![image](https://github.com/user-attachments/assets/877b911d-ca0c-4947-baf2-a6b2237b1825)
