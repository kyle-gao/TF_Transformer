# TF_Transformer

A tensorflow Transformer trained on the Aligned Hansards of the 36th Parliament of Canada: Senate debate training set consisting of 182k english-french sentence pairs.

The data was processed into lowercase and converted from unicode to utf which removed french accented letters.

The tokenizers were given a target vocabulary size of around 16k. 

The token list and data can be found in /Data

Sentences of token length <10 or >50 where filtered out.
