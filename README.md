# **Congressional Tweet DataLoader**

This program can read and load large Congressional Tweet dataset. It will read and load their text and their corresponding party.

## **Description**

- DatasetReader: Used to read data from files and create Instances
- DataLoader: Used to piece together several Instances of data into several Batches for training

### **Construct DatasetReader**

1. The `_read` method, where the input is the path to the file you want to read, keeps yielding Instance data
2. `text_to_instance`, called in the `_read` method, the goal is to create Instance data based on the text data obtained from `_read`
3. `TweetReader` can set `max_instances` to specify the maximum number of instances to be retrieved from the data

### **Construct DatasetLoader**

After constructed Datasetreader, we can use AllenNLP build-in function to construct dataloader. However, before we transfer data into batches, we need to convert a series of tokens recorded in the TextField of the Instance to the corresponding idx.

1. We need to iterate through all Instances to build the vocabulary.
2. Using the dictionary, encode the contents of each Field of each Instance.
    - For `TextField`: we need to convert each Token into its corresponding idx using indexer
    - For `LabelField`: we need to number the labels with the vocabulary
    - For SequenceLabelField, we need to number each content of labels

### **Executing program**

```
python dataloader.py
```

## **Acknowledgments**

Inspiration, code snippets, etc.
* [AllenNLP Turorial](https://zhuanlan.zhihu.com/p/352412971)
* [Dataset](https://uofr-my.sharepoint.com/:x:/g/personal/zxu69_ur_rochester_edu/Ecrs94AZXOFGrUilaHQCECwBJyQmZUjoIlJ1-Z2nSP_hGQ?e=fvyFD9)
