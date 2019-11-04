# Related Projects

[gin-config](https://github.com/google/gin-config) provides a similar solution to `HParams`.
These are the differences:

#### Pros

- `HParams` has much 3x less documentation, 6x less code, and the same amount of tests;
    therefore, HParams is less complex and more thoroughly tested.
- `HParams` publishes it's build status and code coverage.
- `HParams` is focused on Python while instead of a new `gin` language.
- `HParams` is framework independent while `gin-config` prioritizes Tensorflow.
- `HParams` supports type checking.

#### Cons

- `gin-config` has been around for longer and is more mature.
- `gin-config` has baked in support for name-spacing.
- `gin-config` has out of the box support for non-Pythonic configuration files (i.e. `gin`).
- `gin-config` has an example notebook.

#### Same

- Both tools support logging hyperparameters to Tensorboard.
- Both tools support disambiguation of configurable names via modules naming.
- Both tools support breaking up hyperparameters into multiple files.
- Both tools support sharing hyperparameters between multiple configurations.
- Both tools support the command line.
- Both tools support making external classes or functions configurable.
- Both tools support making certain hyperparameters required.
