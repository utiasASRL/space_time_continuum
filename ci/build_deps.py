#!/usr/bin/env python3
"""Builds the C++ dependencies of the spacetime wheel: cibuildwheel's before-all step (pyproject.toml).

Builds and installs Eigen, jsoncpp and GTSAM into a prefix, and builds GTSAM's Python module for
one Python version, which the wheel bundles (SPACETIME_BUNDLE_GTSAM_PY), so `gtsam` and
`spacetime` share one libgtsam. Runs on Linux (in the manylinux container), macOS and Windows,
with nothing but the standard library, cmake, git and a C++ compiler (on Windows, run it from a
Visual Studio developer environment: it builds with Ninja, installed here with pip).

Usage: python ci/build_deps.py <project dir>
Environment:
  CIBW_PYTHON      The Python version the wheel is for (e.g. 3.15). On Linux, GTSAM's module is
                   built with `python<version>`; elsewhere with this interpreter, which must be
                   that version.
  CIBW_JOBS        Upper bound on parallel compiles (default: the CPU count).
  GTSAM_JOBS       Parallel compiles for libgtsam (default: one per GTSAM_GB of available memory).
  GTSAM_GB         Memory per libgtsam compile, in GB (default 2).
  GTSAM_WRAP_JOBS  Parallel compiles for GTSAM's Python wrapper (default: one per GTSAM_WRAP_GB).
  GTSAM_WRAP_GB    Memory per wrapper compile, in GB (default 6: GTSAM's own wheel build uses -j1).

The job counts are what keep a GitHub runner from running out of memory: a GTSAM wrapper
translation unit needs several GB, and a 7 GB macOS runner or a 16 GB Linux runner compiling one
per CPU is killed (exit 137). GTSAM is built without TBB, Boost and -march=native, so the wheel
is portable and self-contained. The versions are the ones spacetime is developed against.
"""

import ctypes
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request

EIGEN_VERSION = "3.4.0"
JSONCPP_VERSION = "1.9.6"
GTSAM_VERSION = "4.3.0"


def deps_dir():
    """Where the dependencies are built and installed; pyproject.toml's [tool.cibuildwheel.<os>] uses the same paths."""
    if sys.platform.startswith("linux"):
        return pathlib.Path("/opt/spacetime-deps")  # in the manylinux container
    if sys.platform == "darwin":
        return pathlib.Path("/tmp/spacetime-deps")
    return pathlib.Path("C:/spacetime-deps")


def available_memory_gb():
    """Memory available for compiling, in GB."""
    if sys.platform.startswith("linux"):
        with open("/proc/meminfo") as meminfo:
            available = next(int(line.split()[1]) * 1024 for line in meminfo if line.startswith("MemAvailable:"))
        # In a container with a memory limit (cgroup v2 or v1), /proc/meminfo still reports the host's.
        for limit_file, usage_file in (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
                                       ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
            try:
                limit = open(limit_file).read().strip()
                if limit != "max" and int(limit) < 2**60:
                    available = min(available, int(limit) - int(open(usage_file).read()))
                break
            except (OSError, ValueError):
                continue
        return available / 2**30
    if sys.platform == "darwin":
        total = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True).stdout)
        return 0.75 * total / 2**30  # macOS reports no "available": leave a quarter for the system
    if sys.platform == "win32":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return status.ullAvailPhys / 2**30
    return 4.0


def jobs_for(gb_per_job, override, max_jobs):
    """Parallel compiles: `override` if set, else one per `gb_per_job` of available memory; within [1, max_jobs]."""
    if os.environ.get(override):
        return max(1, int(os.environ[override]))
    return max(1, min(max_jobs, int(available_memory_gb() // gb_per_job)))


def run(*command, cwd=None):
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], cwd=cwd, check=True)


def download(url, into):
    """Downloads and unpacks the tarball at `url` into `into`; returns the folder it unpacked."""
    archive = into / url.rsplit("/", 1)[-1]
    for attempt in range(1, 4):  # CI downloads fail now and then
        print(f"+ download {url} (attempt {attempt})", flush=True)
        try:
            with urllib.request.urlopen(url, timeout=120) as response, open(archive, "wb") as out:
                shutil.copyfileobj(response, out)
            break
        except OSError as error:
            if attempt == 3:
                raise
            print(f"  failed ({error}); retrying in {10 * attempt} s", flush=True)
            time.sleep(10 * attempt)
    with tarfile.open(archive) as tar:
        top = tar.getnames()[0].split("/")[0]
        safe = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}  # Python 3.12+ (and backports)
        tar.extractall(into, **safe)
    return into / top


def target_python(version):
    """The interpreter GTSAM's Python module is built for."""
    if sys.platform.startswith("linux"):
        found = shutil.which(f"python{version}")  # manylinux: one per CPython version
        if not found:
            sys.exit(f"python{version} not found")
        return found
    running = f"{sys.version_info.major}.{sys.version_info.minor}"
    if version and version != running:
        sys.exit(f"CIBW_PYTHON={version}, but this is Python {running}: run this script with Python {version}")
    return sys.executable


def fix_module_rpaths(module_dir, lib_dir):
    """Points GTSAM's Python module at the installed libgtsam (lib_dir), not the build tree's copy.

    _spacetime links the installed one, so otherwise the wheel repair tool would bundle two copies of
    libgtsam, and GTSAM objects would cross between them. (Windows DLLs are found by name instead.)
    """
    for module in module_dir.glob("*.so"):
        if sys.platform.startswith("linux"):
            run("patchelf", "--remove-rpath", module)  # found through LD_LIBRARY_PATH (pyproject.toml)
        elif sys.platform == "darwin":
            load_commands = subprocess.run(["otool", "-l", module], capture_output=True, text=True, check=True).stdout.splitlines()
            rpaths = [load_commands[i + 2].split()[1] for i, line in enumerate(load_commands) if "LC_RPATH" in line]
            for rpath in rpaths:
                run("install_name_tool", "-delete_rpath", rpath, module)
            run("install_name_tool", "-add_rpath", lib_dir, module)


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    version = os.environ.get("CIBW_PYTHON", "")
    python = target_python(version)
    version = version or f"{sys.version_info.major}.{sys.version_info.minor}"
    root = deps_dir()
    prefix, src, gtsam_build = root / "install", root / "src", root / "gtsam-build"
    src.mkdir(parents=True, exist_ok=True)

    cpus = int(os.environ.get("CIBW_JOBS") or os.cpu_count() or 1)
    gtsam_jobs = jobs_for(float(os.environ.get("GTSAM_GB", 2)), "GTSAM_JOBS", cpus)
    wrap_jobs = jobs_for(float(os.environ.get("GTSAM_WRAP_GB", 6)), "GTSAM_WRAP_JOBS", cpus)
    print(f"Python {python}; {available_memory_gb():.1f} GB available, {cpus} CPUs: libgtsam {gtsam_jobs} jobs, "
          f"wrapper {wrap_jobs} jobs; installing into {prefix}", flush=True)

    run(python, "-m", "pip", "install", "--upgrade", "pip")
    run(python, "-m", "pip", "install", "numpy", "pyparsing")  # GTSAM's wrapper generator
    generator = []
    if sys.platform == "win32":
        # Ninja's -j bounds the compiles (Visual Studio's generator compiles one project's files serially).
        run(sys.executable, "-m", "pip", "install", "ninja")
        generator = ["-G", "Ninja"]
    common = [*generator, "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_INSTALL_PREFIX={prefix}", "-DCMAKE_INSTALL_LIBDIR=lib",
              f"-DCMAKE_PREFIX_PATH={prefix}"]
    gtsam_extra = []
    if sys.platform == "darwin":
        # Installed libraries reference each other through @rpath (e.g. libgtsam -> libmetis-gtsam):
        # give them the prefix as an rpath, so delocate can resolve the chain when it bundles them.
        common.append(f"-DCMAKE_INSTALL_RPATH={prefix / 'lib'}")
        # GTSAM values cross between gtsam's module and spacetime's (gtsam.Values holding a Vector,
        # say), found with dynamic_cast<GenericValue<T>>. With pybind11's default hidden visibility,
        # GenericValue<Eigen type> is hidden, so each module has its own typeinfo, and macOS's libc++
        # compares typeinfo by address: the cast fails. Default visibility lets dyld merge them.
        # (pybind11's own namespace stays hidden.) python/CMakeLists.txt does the same for _spacetime.
        gtsam_extra.append("-DCMAKE_CXX_VISIBILITY_PRESET=default")

    # Eigen 3.4 (header-only; manylinux_2_28's eigen3 is 3.3).
    eigen = download(f"https://gitlab.com/libeigen/eigen/-/archive/{EIGEN_VERSION}/eigen-{EIGEN_VERSION}.tar.gz", src)
    # Eigen looks for a Fortran compiler for its optional BLAS: on Windows it finds MinGW's gfortran,
    # which does not work with MSVC, and the configure fails. Only the headers are needed.
    run("cmake", "-S", eigen, "-B", src / "eigen-build", *common, "-DBUILD_TESTING=OFF", "-DEIGEN_BUILD_DOC=OFF",
        "-DCMAKE_Fortran_COMPILER=NOTFOUND")
    run("cmake", "--build", src / "eigen-build", "--config", "Release", "--target", "install")

    # jsoncpp, as a shared library (the static one is often built without -fPIC).
    jsoncpp = download(f"https://github.com/open-source-parsers/jsoncpp/archive/refs/tags/{JSONCPP_VERSION}.tar.gz", src)
    run("cmake", "-S", jsoncpp, "-B", src / "jsoncpp-build", *common, "-DBUILD_SHARED_LIBS=ON", "-DBUILD_STATIC_LIBS=OFF",
        "-DBUILD_OBJECT_LIBS=OFF", "-DJSONCPP_WITH_TESTS=OFF", "-DJSONCPP_WITH_POST_BUILD_UNITTEST=OFF",
        "-DJSONCPP_WITH_EXAMPLE=OFF")
    run("cmake", "--build", src / "jsoncpp-build", "--config", "Release", "--parallel", gtsam_jobs, "--target", "install")

    # GTSAM, with its Python module for this Python version.
    if not (src / "gtsam").exists():
        run("git", "clone", "--depth", "1", "--branch", GTSAM_VERSION, "https://github.com/borglab/gtsam.git", src / "gtsam")
    run("cmake", "-S", src / "gtsam", "-B", gtsam_build, *common,
        "-DGTSAM_BUILD_TESTS=OFF", "-DGTSAM_BUILD_EXAMPLES_ALWAYS=OFF", "-DGTSAM_BUILD_TIMING_ALWAYS=OFF",
        "-DGTSAM_BUILD_DOCS=OFF", "-DGTSAM_BUILD_UNSTABLE=OFF", "-DGTSAM_UNSTABLE_BUILD_PYTHON=OFF",
        "-DGTSAM_BUILD_WITH_MARCH_NATIVE=OFF", "-DGTSAM_BUILD_WITH_WERROR=OFF", "-DGTSAM_WITH_TBB=OFF",
        "-DGTSAM_USE_BOOST_FEATURES=OFF", "-DGTSAM_ENABLE_BOOST_SERIALIZATION=OFF", "-DGTSAM_USE_SYSTEM_EIGEN=ON",
        "-DGTSAM_USE_QUATERNIONS=OFF", "-DGTSAM_SLOW_BUT_CORRECT_BETWEENFACTOR=ON", "-DGTSAM_SLOW_BUT_CORRECT_EXPMAP=ON",
        "-DGTSAM_BUILD_PYTHON=ON", f"-DGTSAM_PYTHON_VERSION={version}",
        f"-DPYTHON_EXECUTABLE:FILEPATH={python}", f"-DPython_EXECUTABLE={python}", *gtsam_extra)
    run("cmake", "--build", gtsam_build, "--config", "Release", "--parallel", gtsam_jobs, "--target", "install")
    run("cmake", "--build", gtsam_build, "--config", "Release", "--parallel", wrap_jobs, "--target", "gtsam_py")

    module = gtsam_build / "python" / "gtsam"
    if not (module / "__init__.py").exists():
        sys.exit(f"GTSAM's Python module was not built in {module}")
    fix_module_rpaths(module, prefix / "lib")
    print("GTSAM's Python module:", sorted(p.name for p in module.iterdir() if p.suffix in (".so", ".pyd")), flush=True)


if __name__ == "__main__":
    main()
