#
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

# pylint: disable=protected-access
from __future__ import absolute_import, division, print_function

import logging
import threading

from kafka import KafkaConsumer
from kafka import KafkaProducer
from pyspark.ml import Estimator

from sparkdl.param import (
    keyword_only, HasLabelCol, HasInputCol, HasOutputCol)
from sparkdl.param.shared_params import HasMapFun, KafkaParam, FitParam
import sparkdl.utils.jvmapi as JVMAPI

__all__ = ['TFTextFileEstimator']

logger = logging.getLogger('sparkdl')


class TFTextFileEstimator(Estimator, HasInputCol, HasOutputCol, HasLabelCol, HasMapFun, KafkaParam, FitParam):
    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, labelCol=None, mapFun=None, kafkaParam=None, fitParam=None):
        # NOTE(phi-dbq): currently we ignore output mode, as the actual output are the
        #                trained models and the Transformers built from them.
        super(TFTextFileEstimator, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, labelCol=None, mapFun=None, kafkaParam=None, fitParam=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def fit(self, dataset, params=None):
        self._validateParams()
        if params is None:
            paramMaps = [dict()]
        elif isinstance(params, (list, tuple)):
            if len(params) == 0:
                paramMaps = [dict()]
            else:
                self._validateFitParams(params)
                paramMaps = params
        elif isinstance(params, dict):
            self._validateFitParams(params)
            paramMaps = [params]
        else:
            raise ValueError("Params must be either a param map or a list/tuple of param maps, "
                             "but got %s." % type(params))
        print(paramMaps)
        return self._fitInParallel(dataset, paramMaps)

    def _validateParams(self):
        """
        Check Param values so we can throw errors on the driver, rather than workers.
        :return: True if parameters are valid
        """
        if not self.isDefined(self.inputCol):
            raise ValueError("Input column must be defined")
        if not self.isDefined(self.outputCol):
            raise ValueError("Output column must be defined")
        return True

    def _validateFitParams(self, params):
        """ Check if an input parameter set is valid """
        if isinstance(params, (list, tuple, dict)):
            assert self.getInputCol() not in params, \
                "params {} cannot contain input column name {}".format(params, self.getInputCol())
        return True

    def _fitInParallel(self, dataset, paramMaps):
        """
        Fits len(paramMaps) models in parallel, one in each Spark task.
        :param paramMaps: non-empty list or tuple of ParamMaps (dict values)
        :return: list of fitted models, matching the order of paramMaps
        """

        def _write_data():
            def _write_partition(d_iter):
                producer = KafkaProducer(bootstrap_servers=self.getKafkaParam()["host"])
                for d in d_iter:
                    producer.send(self.getKafkaParam()["topic"], d)
                producer.send(self.getKafkaParam()["topic"], "_stop_")
                producer.flush()
                producer.close()
                dataset.rdd.foreachPartition(lambda p: _write_partition(p))

        t = threading.Thread(target=_write_data)
        t.start()
        stop_flag_num = len(dataset.rdd.partitions)
        vacab_size = dataset.take(1)[0]["vocab_size"]

        sc = JVMAPI._curr_sc()
        paramMapsRDD = sc.parallelize(paramMaps, numSlices=len(paramMaps))

        # Obtain params for this estimator instance
        baseParamMap = self.extractParamMap()
        baseParamDict = dict([(param.name, val) for param, val in baseParamMap.items()])
        baseParamDictBc = sc.broadcast(baseParamDict)

        def _local_fit(override_param_map):
            # Update params
            params = baseParamDictBc.value
            override_param_dict = dict([
                                           (param.name, val) for param, val in override_param_map.items()])
            params.update(override_param_dict)

            def _read_data():
                consumer = KafkaConsumer(self.getKafkaParam()["topic"],
                                         group_id=self.getKafkaParam()["group_id"],
                                         bootstrap_servers=self.getKafkaParam()["host"]
                                         )
                stop_count = 0
                while True:
                    messages = consumer.poll(timeout_ms=1000, max_records=64)
                    group_msgs = []
                    for tp, message in messages.items():
                        msg_value = message.value.decode()
                        if msg_value == "_stop_":
                            stop_count += 1
                        else:
                            group_msgs.append(msg_value)

                    yield group_msgs
                    if stop_count == stop_flag_num:
                        break

                consumer.close()

            self.mapFun(_read_data,
                        feature=self.getInputCol(),
                        label=self.getLabelCol(),
                        vacab_size=vacab_size
                        )

        paramMapsRDD.map(lambda paramMap: (paramMap, _local_fit(paramMap)))

    def _fit(self, dataset):  # pylint: disable=unused-argument
        err_msgs = ["This function should not have been called",
                    "Please contact library maintainers to file a bug"]
        raise NotImplementedError('\n'.join(err_msgs))
