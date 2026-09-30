"""Bundle a supplied GGUF with checksum-pinned Mozilla inference binaries."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import httpx

VERSION = "0.10.6"
ASSETS = {
    "llamafile-0.10.6-thin": "756bf97febae557084e8ced9bde7a1cb0c7f6ce778530d582afa219406237a50",
    "zipalign-0.10.6": "7eac59c658226027b365d92131babf06341f746fb20c0db33115bbdf5c5c6c4b",
}


def checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(directory, name, digest):
    target = directory / name
    url = f"https://github.com/mozilla-ai/llamafile/releases/download/{VERSION}/{name}"
    if not target.exists():
        partial = target.with_suffix(".partial")
        with httpx.Client(timeout=180, follow_redirects=True) as client:
            with client.stream("GET", url) as response, partial.open("wb") as stream:
                response.raise_for_status()
                for chunk in response.iter_bytes():
                    stream.write(chunk)
        partial.rename(target)
    if checksum(target) != digest:
        raise ValueError(f"Checksum mismatch for {name}; remove the cached file and retry")
    target.chmod(0o755)
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--output", default="dist/trading-language.llamafile")
    parser.add_argument("--cache", default="artifacts/llamafile")
    parser.add_argument("--loader", help="Optional APE loader path on hosts without APE support")
    parser.add_argument("--model-notice", help="Model attribution/license notice to embed")
    args = parser.parse_args()
    gguf = Path(args.gguf).resolve()
    output = Path(args.output).resolve()
    cache = Path(args.cache).resolve()
    if output.exists():
        raise SystemExit("Output already exists; choose a new filename")
    with gguf.open("rb") as stream:
        if stream.read(4) != b"GGUF":
            raise SystemExit("Input is not a GGUF file")
    cache.mkdir(parents=True, exist_ok=True)
    binary, zipper = [download(cache, name, digest) for name, digest in ASSETS.items()]
    output.parent.mkdir(parents=True, exist_ok=True)
    flags = cache / ".args"
    flags.write_text(
        f"-m\n/zip/{gguf.name}\n--server\n--host\n127.0.0.1\n--port\n8080\n"
        "--ctx-size\n4096\n--threads\n4\n--no-mmap\n...\n"
    )
    shutil.copyfile(binary, output)
    license_dir = Path(__file__).resolve().parents[1] / "docs/licenses"
    notices = [license_dir / "Apache-2.0.txt", license_dir / "llamafile-LICENSE.txt"]
    if args.model_notice:
        notices.append(Path(args.model_notice).resolve())
    try:
        command = [str(zipper), "-j0", str(output), str(gguf), str(flags)] + [
            str(path) for path in notices
        ]
        if args.loader:
            command.insert(0, str(Path(args.loader).resolve()))
        subprocess.run(command, check=True)
    except (subprocess.CalledProcessError, OSError):
        output.unlink(missing_ok=True)
        raise
    output.chmod(0o755)
    manifest = {
        "llamafile_version": VERSION,
        "upstream_binaries": ASSETS,
        "gguf_sha256": checksum(gguf),
        "executable_sha256": checksum(output),
        "bytes": output.stat().st_size,
        "license_files": [path.name for path in notices],
        "windows_under_4gb": output.stat().st_size < 4 * 1024**3,
        "model_license": "Supply the upstream model license and notices when redistributing",
    }
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"output": str(output), **manifest}, indent=2))


if __name__ == "__main__":
    main()
