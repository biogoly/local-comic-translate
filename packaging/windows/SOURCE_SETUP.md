# Local Comic Translate for Windows x64

1. Install **64-bit Python 3.12** for your user. The [Python 3.12.10 Windows release](https://www.python.org/downloads/release/python-31210/) provides a Windows installer. Include the Python Launcher and pip. If you already use [uv](https://docs.astral.sh/uv/getting-started/installation/), you can instead run `uv python install 3.12`.
2. Extract the **whole ZIP** into a short folder you own, such as your Downloads folder. Open the extracted `LCT` folder and double-click **run.bat**.
3. On first launch, leave the console open while the pinned Python packages download from PyPI. The Windows GPU libraries need several GB of disk space. The launcher checks dependency hashes, installs into a dedicated environment under `%LOCALAPPDATA%\LocalComicTranslate\SourceRuntime`, and opens the editor. Later launches reuse that environment. You can use CPU processing without an NVIDIA GPU.

The ZIP contains the editor's source, UI resources, and their notices. The included Dayu widget code is credited in `packaging/windows/DAYU_LICENSE.txt`. The Python dependencies and model weights are obtained separately. Optional model-based tools follow the editor's own model setup. External ComfyUI and FLUX installations use their own environments.

**If setup fails**, read the console message. Verify that Python 3.12 x64 is available and check your network and disk space, then rerun `run.bat`. With the standard Python install, `py -3.12-64 --version` should identify Python 3.12. The launcher can use either uv or standard Python's venv and pip. Administration rights are not required.

**To prepare the environment without opening the app**, run `run.bat -SetupOnly` in the extracted folder. To retry installation into an existing runtime, use `run.bat -Repair -SetupOnly`. To choose the venv/pip setup explicitly, use `run.bat -UsePip -SetupOnly`. You can also pass a project path directly to `run.bat`.

**To remove this installation**, close the editor, then remove its extracted source folder. The isolated runtime and its installer cache can be removed from the `SourceRuntime` folder above. Keep your project files, models, and editor profile according to your own preferences.

`SOURCE-INFO.json` identifies the source commit and includes checksums for the release files.
