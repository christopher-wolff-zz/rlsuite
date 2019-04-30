# Tests

The tests are based on the `pytest` package. To run them, simple execute

```bash
pytest .
```

from the project's root directory.

The tests make sure that none of the algorithms break when called with their default parameters. In the future, we plan to add additional checks based on performance metrics, e.g. achieving a certain RMSE on a simple test task.

All algorithms have been tested on sample environments and should work as intended. However, if you encounter a bug, please submit an issue [here](https://github.com/christopher-wolff/rlsuite/issues).
