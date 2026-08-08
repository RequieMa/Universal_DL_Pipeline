# Data

## CsvDataSource

::: pipeline.data.csv_source
    options:
      members:
        - CsvDataSource

## train_test_split

::: pipeline.data.split
    options:
      members:
        - train_test_split

## ImageFolderDataSource

::: pipeline.data.image_folder
    options:
      members:
        - ImageFolderDataSource

## TransformedDataStream

::: pipeline.data.transforms
    options:
      members:
        - TransformedDataStream

## TextDataSource

::: pipeline.data.text_source
    options:
      members:
        - TextDataSource

## collect_arrays

::: pipeline.data.utils
    options:
      members:
        - collect_arrays
