# import packages
import pandas as pd
from typing import Dict
from allennlp.data import DatasetReader, Tokenizer, TokenIndexer, Instance, Vocabulary
from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WhitespaceTokenizer


class TweetReader(DatasetReader):
    # initialize DatasetReader
    def __init__(self,
                 max_instances=600_000,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_tokens: int = None):
        super().__init__(max_instances=max_instances)
        # If tokenizer not provided: WhitespaceTokenizer
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        # If token_indexers not provided: SingleIdTokenIndexer
        self.token_indexers = token_indexers or {
            'tokens': SingleIdTokenIndexer()}
        self.max_tokens = max_tokens

    # build instances
    def text_to_instance(self, fields):
        return Instance(fields)

    # read data
    def _read(self, file_path: str):
        df = pd.read_csv(file_path, index_col=0)
        for _, row in df.iterrows():
            text = row["full_text"]
            tokens = self.tokenizer.tokenize(text)
            if self.max_tokens:
                tokens = tokens[:self.max_tokens]
            text_field = TextField(tokens, self.token_indexers)
            label_field = LabelField(row["party_id"])
            fields = {'text': text_field, 'label': label_field}
            yield self.text_to_instance(fields)


if __name__ == '__main__':
    # define a dataset reader
    reader = TweetReader()
    data_path = "congressional_tweet.csv"
    # build data loader
    dataloader = MultiProcessDataLoader(reader, data_path, batch_size=2)
    # Call the iter_instances method of the dataloader to get the generator of instances
    instances = dataloader.iter_instances()
    # Call the constructor of Vocabulary
    vocab = Vocabulary.from_instances(instances)
    # Using vocab, get the index of each Field of all Instances in the dataset
    dataloader.index_with(vocab)

    # Call collate_fn of dataloader to organize the data into a Batch
    for batch in dataloader:
        print(batch)
        break
