# clusterizer
My first attempts to clusterize texts.

The sequence of using scripts in order to get results is presented on the picture below.

![scheme](https://github.com/zeionara/clusterizer/blob/master/docs/scheme.jpg)

Example of usage of the presented scripts for splitting datasets into clusters and generating reports:

```sh
python3 -m clusterizer.lemmatize -in /mnt/c/Users/prote/big_data/news_lenta.csv -out /mnt/c/Users/prote/big_data/test.csv -i 1 -w 700000 -s 0.05 -n 5000 -v && python3 -m clusterizer.filter -in /mnt/c/Users/prote/big_data/russian_news.txt -out /mnt/c/Users/prote/big_data/test.txt -i 1 -w 1500000 -s 0.05 -n 10000 -v -k natural_disasters_keywords.txt -c && 
python3 -m clusterizer.filter -in /mnt/c/Users/prote/big_data/test.csv -out /mnt/c/Users/prote/big_data/test.txt -i 1 -w 700000 -s 0.05 -v -k natural_disasters_keywords.txt && 
python3 -m clusterizer.erase -in /mnt/c/Users/prote/big_data/test.txt -out /mnt/c/Users/prote/big_data/twsw.txt -v &&
python3 -m clusterizer.randomize -in /mnt/c/Users/prote/big_data/twsw.txt -out /mnt/c/Users/prote/big_data/clustering_results/articles_for_reducing_vocabulary.txt -n 200 -v && 
python3 -m clusterizer.reduce -in /mnt/c/Users/prote/big_data/clustering_results/articles_for_reducing_vocabulary.txt -out /mnt/c/Users/prote/big_data/clustering_results/articles_for_clustering.txt -v && 
python3 -m clusterizer.clusterize -min 2 -max 10 -i 100 -in /mnt/c/Users/prote/big_data/clustering_results/articles_for_clustering.txt -ofo /mnt/c/Users/prote/big_data/clustering_results/results/results -ofi cluster -rfi /mnt/c/Users/prote/big_data/clustering_results/reports/results -d 100 -v -e -s &&
python3 -m clusterizer.format -in /mnt/c/Users/prote/big_data/clustering_results/results/results_100 -out /mnt/c/Users/prote/big_data/clustering_results/reports/clusters_descriptions.txt -n 5 -v
```

Plot, built on the data from the output report generated by the command given above:

![average silhouette](https://github.com/zeionara/clusterizer/blob/master/docs/avg_slh.jpg)
