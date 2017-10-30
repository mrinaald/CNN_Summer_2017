# Description: Tensorflow Serving examples.

package(
    default_visibility = ["//tensorflow_serving:internal"],
    features = ["no_layering_check"],
)

licenses(["notice"])  # Apache 2.0

load("//tensorflow_serving:serving.bzl", "serving_proto_library")

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
)

py_binary(
    name = "mobilenet_client",
    srcs = [
        "mobilenet_client.py",
    ],
    deps = [
        "//tensorflow_serving/apis:predict_proto_py_pb2",
        "//tensorflow_serving/apis:prediction_service_proto_py_pb2",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)