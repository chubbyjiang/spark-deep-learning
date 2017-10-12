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
from sparkdl.estimators.tf_text_file_estimator import TFTextFileEstimator
from sparkdl.transformers.tf_text import TFTextTransformer
from sparkdl.tf_fun import map_fun
from ..tests import SparkDLTestCase


class TFTextTransformerTest(SparkDLTestCase):
    def test_loadText(self):
        input_col = "text"
        output_col = "sentence_matrix"

        documentDF = self.session.createDataFrame([
            ("Hi I heard about Spark", 1),
            ("I wish Java could use case classes", 0),
            ("Logistic regression models are neat", 2)
        ], ["text", "preds"])

        transformer = TFTextTransformer(
            inputCol=input_col, outputCol=output_col)

        df = transformer.transform(documentDF)
        estimator = TFTextFileEstimator(inputCol="sentence_matrix", outputCol="sentence_matrix", labelCol="preds",
                                        kafkaParam={"host": "127.0.0.1", "topic": "test", "group_id": "sdl_1"},
                                        fitParam=[{"epochs": 5, "batch_size": 64}, {"epochs": 5, "batch_size": 1}],
                                        mapFnParam=map_fun)
        estimator.fit(df).collect()
