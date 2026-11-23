# Read and write Parquet files in batches. Every step uses these, so only one batch is in memory at a time.

import pyarrow as pa
import pyarrow.parquet as pq

import config


# Yield a Parquet file batch by batch, each batch is a list of dicts.
def read_batches(path, columns=None):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=config.BATCH_SIZE, columns=columns):
        yield batch.to_pylist()


# Yield a Parquet file one article at a time (read in batches under the hood).
def read_rows(path, columns=None):
    for batch in read_batches(path, columns):
        yield from batch


# Open a Parquet file for writing batch by batch; every column is a string.
def open_writer(path, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([(name, pa.string()) for name in columns])
    return pq.ParquetWriter(path, schema, compression="zstd")


# Write one batch (list of dicts) with an open writer.
def write_batch(writer, rows):
    if rows:
        writer.write_table(pa.Table.from_pylist(rows, schema=writer.schema))
