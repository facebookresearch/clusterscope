import getpass
import os
import socket
from typing import Dict, Hashable, Optional

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.metrics import NoOpMeterProvider
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import NoOpTracerProvider

from clusterscope.lib import (
    cluster,
    global_rank,
    job_id,
    job_name,
    local_node_gpu_generation_and_count,
    local_rank,
    world_size,
)


OTEL_EXPORTER_OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")


def setup_opentelemetry(
    resource_attributes: Optional[Dict[Hashable, str | int]] = None,
    init_tracer_provider: bool = True,
    init_meter_provider: bool = True,
):
    """Set up OpenTelemetry SDK with OTLP HTTP exporters."""
    final_resource_attributes = {
        "job_id": job_id,
        "job_name": job_name,
        "user": getpass.getuser(),
        "cluster": cluster(),
        "local_rank": local_rank,
        "global_rank": global_rank,
        "world_size": world_size,
        "host": socket.gethostname(),
        "gpu_type": local_node_gpu_generation_and_count(),
        **(resource_attributes or {}),
    }
    resource = Resource(attributes=resource_attributes)

    if init_tracer_provider:
        # Set up tracing with OTLP HTTP exporter
        tracer_provider = TracerProvider(resource=resource)
        otlp_span_exporter = OTLPSpanExporter(
            endpoint=OTEL_EXPORTER_OTLP_ENDPOINT + "/v1/traces", timeout=60
        )
        span_processor = BatchSpanProcessor(otlp_span_exporter)
        tracer_provider.add_span_processor(span_processor)
    else:
        tracer_provider = NoOpTracerProvider()
    trace.set_tracer_provider(tracer_provider)

    if init_meter_provider:
        # Set up metrics with OTLP HTTP exporter
        otlp_metric_exporter = OTLPMetricExporter(
            endpoint=OTEL_EXPORTER_OTLP_ENDPOINT + "/v1/metrics", timeout=60
        )
        metric_reader = PeriodicExportingMetricReader(
            otlp_metric_exporter,
            export_interval_millis=60000,
            export_timeout_millis=30000,
        )
        meter_provider = MeterProvider(
            resource=resource, metric_readers=[metric_reader], shutdown_on_exit=True
        )
    else:
        meter_provider = NoOpMeterProvider()
    metrics.set_meter_provider(meter_provider)

    return tracer_provider, meter_provider
