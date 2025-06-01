import csv
import os
import numpy as np

class CSVDataWriter:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file_handle = None
        self.writer = None
        self._open_file()

    def _open_file(self):
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            self.file_handle = open(self.file_path, mode='w', newline='', encoding='utf-8')
            self.writer = csv.writer(self.file_handle)
            print(f"CSV file opened for writing: {self.file_path}")
        except IOError as e:
            print(f"Error opening CSV file {self.file_path}: {e}")
            self.file_handle = None
            self.writer = None
        except Exception as e:
            print(f"An unexpected error occurred while opening CSV file {self.file_path}: {e}")
            self.file_handle = None
            self.writer = None

    def write_header(self, header_list: list[str]):
        if self.writer:
            try:
                self.writer.writerow(header_list)
                print(f"CSV header written to {self.file_path}")
            except Exception as e:
                print(f"Error writing CSV header to {self.file_path}: {e}")
        else:
            print(f"Error: CSV writer not initialized. Cannot write header to {self.file_path}.")

    def write_row(self, data_row: list):
        if self.writer:
            try:
                # Replace potential NaN values with empty strings or a specific placeholder
                processed_row = ["" if isinstance(x, float) and np.isnan(x) else x for x in data_row]
                self.writer.writerow(processed_row)
            except Exception as e:
                print(f"Error writing CSV row to {self.file_path}: {e}")
        # else: # Avoid console spam if writer failed to initialize
            # print(f"Error: CSV writer not initialized. Cannot write row to {self.file_path}")


    def close_file(self):
        if self.file_handle:
            try:
                self.file_handle.close()
                print(f"CSV file closed: {self.file_path}")
            except Exception as e:
                print(f"Error closing CSV file {self.file_path}: {e}")
        self.file_handle = None
        self.writer = None