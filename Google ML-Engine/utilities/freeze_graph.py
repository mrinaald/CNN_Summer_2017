# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Freezes a tensorflow graph using its checkpoint file and graph.pb/graph.pbtxt files."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.tools import freeze_graph

# Changing the directory to make the relative paths of the input files smaller
os.chdir('/Users/mrinaldogra/Downloads/Ubuntu/Project/Google Cloud ML/temp/v3.2/training/')

# We save out the graph to disk, and then call the const conversion
# routine.
input_graph_path = 'model/mrinaal.pbtxt'        # The graph file written using the tf.train.write_graph()
input_saver_def_path = ""
input_binary = False                            # True if the input_graph_path file is a .pb file rather than .pbtxt
checkpoint_path = 'model/mrinaal.ckpt'          # The checkpoint file saved
output_node_names = "ArgMax"                    # Name of the output nodes
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_graph_path = 'freeze_mrinaal.pb'         # Name of the output graph
clear_devices = False

freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                          input_binary, checkpoint_path, output_node_names,
                          restore_op_name, filename_tensor_name,
                          output_graph_path, clear_devices, "")
