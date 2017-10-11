# Copyright 2017 Databricks, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import numpy as np
from pyspark.ml import Transformer
from pyspark.ml.feature import Word2Vec
from pyspark.sql.functions import udf
from pyspark.sql import functions as f
from pyspark.sql.types import *
from pyspark.sql.functions import lit
from sparkdl.param import (
    keyword_only, HasInputCol, HasOutputCol)
import re

import sparkdl.utils.jvmapi as JVMAPI


class TFTextTransformer(Transformer, HasInputCol, HasOutputCol):
    """
    Applies the Tensorflow graph to the image column in DataFrame.

    Restrictions of the current API:

    * Does not use minibatches, which is a major low-hanging fruit for performance.
    * Only one output node can be specified.
    * The output is expected to be an image or a 1-d vector.
    * All images in the dataframe are expected be of the same numerical data type
      (i.e. the dtype of the values in the numpy array representation is the same.)

    We assume all graphs have a "minibatch" dimension (i.e. an unknown leading
    dimension) in the tensor shapes.

    .. note:: The input tensorflow graph should have appropriate weights constantified,
              since a new session is created inside this transformer.
    """

    USER_GRAPH_NAMESPACE = 'given'
    NEW_OUTPUT_PREFIX = 'sdl_flattened'

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, word_dim=100):
        """
        __init__(self, inputCol=None, outputCol=None, graph=None,
                 inputTensor=utils.IMAGE_INPUT_PLACEHOLDER_NAME, outputTensor=None,
                 outputMode="vector")
        """
        super(TFTextTransformer, self).__init__()
        self.word_dim = word_dim
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, word_dim=100):
        """
        setParams(self, inputCol=None, outputCol=None, graph=None,
                  inputTensor=utils.IMAGE_INPUT_PLACEHOLDER_NAME, outputTensor=None,
                  outputMode="vector")
        """
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        """
        :param dataset: dataset columns include: id,content,pred . Make sure words in content column is
         separated by white space.
        :return: new dataset
        """
        word2vec = Word2Vec(vectorSize=self.word_dim, minCount=1, inputCol=self.getInputCol(),
                            outputCol="word_embedding")
        word_embedding = dict(
            word2vec.fit(
                dataset.select(f.split(self.getInputCol(), "\\s+").alias(self.getInputCol()))).getVectors().rdd.map(
                lambda p: (p.word, p.vector.values.tolist())).collect())
        word_embedding["unk"] = np.zeros(100).tolist()
        sc = JVMAPI._curr_sc()
        local_word_embedding = sc.broadcast(word_embedding)

        def convert_word_to_index(s):
            def _pad_sequences(sequences, maxlen=None):
                new_sequences = []

                if len(sequences) <= maxlen:
                    for i in range(maxlen - len(sequences)):
                        new_sequences.append(np.zeros(self.word_dim).tolist())
                    return sequences + new_sequences
                else:
                    return sequences[0:maxlen]

            new_q = [local_word_embedding.value[word] for word in re.split(r"\s+", s) if
                     word in local_word_embedding.value.keys()]
            result = _pad_sequences(new_q, maxlen=64)
            return result

        cwti_udf = udf(convert_word_to_index, ArrayType(ArrayType(FloatType())))
        doc_martic = (dataset.withColumn(self.getOutputCol(), cwti_udf(self.getInputCol()).alias(self.getOutputCol()))
                      .withColumn("vocab_size", lit(len(word_embedding)))
                      )

        return doc_martic
