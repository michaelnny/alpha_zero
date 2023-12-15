# Copyright (c) 2023 Michael Hu.
# This code is part of the book "The Art of Reinforcement Learning: Fundamentals, Mathematics, and Implementation with Python.".
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.


"""A simple class to write statistics to csv file."""
import os
import csv
import time


class CsvWriter:
    """A logging object writing to a CSV file."""

    def __init__(self, fname, buffer_size=100, flush_interval=60):
        """Initializes a `CsvWriter`.
        Args:
          fname: File name (path) for file to be written to.
          buffer_size: Number of rows to buffer before writing to disk.
          flush_interval: Time (in seconds) to wait before writing buffered rows
                          to disk, even if the buffer hasn't reached `buffer_size`.
        """
        dirname = os.path.dirname(fname)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        self._fname = fname
        self._fieldnames = None
        self._header_written = False if self.check_is_empty() else True

        self._buffer = []
        self._buffer_size = buffer_size
        self._flush_interval = flush_interval
        self._last_flush_time = time.time()

    def write(self, values):
        """Appends given values as new row to CSV file."""
        if self._fieldnames is None:
            self._fieldnames = values.keys()

        self._buffer.append(values)

        if len(self._buffer) >= self._buffer_size or time.time() - self._last_flush_time >= self._flush_interval:
            self._flush_buffer()

    def check_is_empty(self):
        # Check if already has rows
        empty = True
        if os.path.exists(self._fname):
            with open(self._fname, 'r', encoding='utf8') as csv_file:
                content = csv.reader(csv_file)
                if len(list(content)) > 0:
                    empty = False
        return empty

    def close(self):
        """Closes the `CsvWriter`."""
        self._flush_buffer()

    def _flush_buffer(self):
        """Writes buffered rows to disk and clears the buffer."""
        if not self._buffer:
            return

        # Open a file in 'append' mode, so we can continue logging safely to the
        # same file after e.g. restarting from a checkpoint.
        with open(self._fname, 'a') as file:
            # Always use same fieldnames to create writer, this way a consistency
            # check is performed automatically on each write.
            writer = csv.DictWriter(file, fieldnames=self._fieldnames)
            # Write a header if this is the very first write.
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerows(self._buffer)
            self._buffer.clear()

        self._last_flush_time = time.time()
