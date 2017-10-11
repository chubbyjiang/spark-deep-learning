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

import cPickle as pickle

from kafka import KafkaConsumer
from kafka import KafkaProducer
from pyspark.ml import Estimator

from sparkdl.param import (
    keyword_only, HasLabelCol, HasInputCol, HasOutputCol)
from sparkdl.param.shared_params import KafkaParam, FitParam
import sparkdl.utils.jvmapi as JVMAPI

__all__ = ['TFTextFileEstimator']

logger = logging.getLogger('sparkdl')


class TFTextFileEstimator(Estimator, HasInputCol, HasOutputCol, HasLabelCol, KafkaParam, FitParam):
    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, labelCol=None, kafkaParam=None, fitParam=None):
        # NOTE(phi-dbq): currently we ignore output mode, as the actual output are the
        #                trained models and the Transformers built from them.
        super(TFTextFileEstimator, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None, labelCol=None, kafkaParam=None, fitParam=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def mapFun(self, _mapFun):
        self._mapFun = _mapFun

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

        inputCol = self.getInputCol()
        labelCol = self.getLabelCol()
        from time import gmtime, strftime
        topic = self.getKafkaParam()["topic"] + "_" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        group_id = self.getKafkaParam()["group_id"]
        host = self.getKafkaParam()["host"]

        def _write_data():
            def _write_partition(d_iter):
                producer = KafkaProducer(bootstrap_servers=host)
                for d in d_iter:
                    producer.send(topic, pickle.dumps(d))
                producer.send(topic, pickle.dumps("_stop_"))
                producer.flush()
                producer.close()

            dataset.rdd.foreachPartition(lambda p: _write_partition(p))

        t = threading.Thread(target=_write_data)
        t.start()
        stop_flag_num = dataset.rdd.getNumPartitions()
        temp_item = dataset.take(1)[0]
        vocab_s = temp_item["vocab_size"]
        embedding_size = temp_item["embedding_size"]

        sc = JVMAPI._curr_sc()
        paramMapsRDD = sc.parallelize(paramMaps, numSlices=len(paramMaps))
        print("paramMaps {}; paramMapsRDD {}".format(paramMaps, paramMapsRDD.count()))

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

            def _read_data(max_records=64):
                consumer = KafkaConsumer(topic,
                                         group_id=group_id,
                                         bootstrap_servers=host,
                                         auto_offset_reset="earliest",
                                         enable_auto_commit=False
                                         )
                stop_count = 0
                fail_msg_count = 0
                while True:
                    messages = consumer.poll(timeout_ms=1000, max_records=max_records)
                    group_msgs = []
                    for tp, records in messages.items():
                        for record in records:
                            try:
                                msg_value = pickle.loads(record.value)
                                if msg_value == "_stop_":
                                    stop_count += 1
                                else:
                                    group_msgs.append(msg_value)
                            except:
                                fail_msg_count += 0
                                pass
                    if len(group_msgs) > 0:
                        yield group_msgs
                    # print("stop_count = {} group_msgs = {} stop_flag_num = {} fail_msg_count = {}".format(stop_count,
                    #                                                                                       len(
                    #                                                                                           group_msgs),
                    #                                                                                       stop_flag_num,
                    #                                                                                       fail_msg_count))

                    if stop_count >= stop_flag_num and len(group_msgs) == 0:
                        break

                consumer.close()

            self._mapFun(_read_data,
                         feature=inputCol,
                         label=labelCol,
                         vacab_size=vocab_s,
                         embedding_size=embedding_size
                         )

        return paramMapsRDD.map(lambda paramMap: (paramMap, _local_fit(paramMap)))

    def _fit(self, dataset):  # pylint: disable=unused-argument
        err_msgs = ["This function should not have been called",
                    "Please contact library maintainers to file a bug"]
        raise NotImplementedError('\n'.join(err_msgs))
